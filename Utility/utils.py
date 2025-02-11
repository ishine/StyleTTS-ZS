from monotonic_align.core import maximum_path_c
import numpy as np
import torch
import torch.nn.functional as F
import os
from functools import reduce
from inspect import isfunction
from math import ceil, floor, log2
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union
from torch import Tensor
from typing_extensions import TypeGuard

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def maximum_path(neg_cent, mask):
  """ Cython optimized version.
  neg_cent: [b, t_t, t_s]
  mask: [b, t_t, t_s]
  """
  device = neg_cent.device
  dtype = neg_cent.dtype
  neg_cent =  np.ascontiguousarray(neg_cent.data.cpu().numpy().astype(np.float32))
  path =  np.ascontiguousarray(np.zeros(neg_cent.shape, dtype=np.int32))

  t_t_max = np.ascontiguousarray(mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32))
  t_s_max = np.ascontiguousarray(mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32))
  maximum_path_c(path, neg_cent, t_t_max, t_s_max)
  return torch.from_numpy(path).to(device=device, dtype=dtype)


def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data_LibriTTS_R_comb/train_list_new.txt"
    if val_path is None:
        val_path = "Data_LibriTTS_R_comb/val_list_new.txt"

    with open(train_path, 'r') as f:
        train_list = f.readlines()
    with open(val_path, 'r') as f:
        val_list = f.readlines()

    # train_list = train_list[-1000:]
    # val_list = train_list[:1000]
    return train_list, val_list


def find_adjacent_file(file_path):
    # Extract the directory and file name
    directory, current_file = os.path.split(file_path)
    # List all files in the directory
    files = sorted([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.wav')])
        
    # Find the index of the current file
    try:
        current_index = files.index(current_file)
    except ValueError:
        return None  # Current file not found in the directory
    
    nums = current_file.split('.')[0].split('_')
    passage = int(nums[-2])
    sentence = int(nums[-1])
    
    new_index = current_index - 1
    # Check if the new index is within the bounds of the list
    if 0 <= new_index < len(files):
        prev_file = os.path.join(directory, files[new_index])
        nums = files[new_index].split('.')[0].split('_')
        prev_passage = int(nums[-2])
        prev_sentence = int(nums[-1])
        
        prev_dist = np.abs(prev_passage - passage) * 10 + np.abs(prev_sentence - sentence)
    else:
        prev_file = None  # No adjacent file (next or previous) found
        prev_dist = np.inf
        
    new_index = current_index + 1
    # Check if the new index is within the bounds of the list
    if 0 <= new_index < len(files):
        next_file = os.path.join(directory, files[new_index])
        
        nums = files[new_index].split('.')[0].split('_')
        prev_passage = int(nums[-2])
        prev_sentence = int(nums[-1])
        
        next_dist = np.abs(prev_passage - passage) * 10 + np.abs(prev_sentence - sentence)
        
    else:
        next_file = None  # No adjacent file (next or previous) found
        next_dist = np.inf
    
    if prev_dist <= next_dist:
        return prev_file
    else:
        return next_file
    

def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window,
            return_complex=True)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.abs(x_stft).transpose(2, 1)


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

# for adversarial loss
def adv_loss(logits, target):
    assert target in [1, 0]
    if len(logits.shape) > 1:
        logits = logits.reshape(-1)
    targets = torch.full_like(logits, fill_value=target)
    logits = logits.clamp(min=-10, max=10) # prevent nan
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss

# for R1 regularization loss
def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg

# for F0 consistency loss
def compute_mean_f0(f0):
    f0_mean = f0.mean(-1)
    f0_mean = f0_mean.expand(f0.shape[-1], f0_mean.shape[0]).transpose(0, 1) # (B, M)
    return f0_mean

def f0_loss(x_f0, y_f0):
    """
    x.shape = (B, 1, M, L): predict
    y.shape = (B, 1, M, L): target
    """
    # compute the mean
    x_mean = compute_mean_f0(x_f0)
    y_mean = compute_mean_f0(y_f0)
    loss = F.l1_loss(x_f0 / x_mean, y_f0 / y_mean)
    return loss

# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x

T = TypeVar("T")


def exists(val: Optional[T]) -> TypeGuard[T]:
    return val is not None


def iff(condition: bool, value: T) -> Optional[T]:
    return value if condition else None


def is_sequence(obj: T) -> TypeGuard[Union[list, tuple]]:
    return isinstance(obj, list) or isinstance(obj, tuple)


def default(val: Optional[T], d: Union[Callable[..., T], T]) -> T:
    if exists(val):
        return val
    return d() if isfunction(d) else d


def to_list(val: Union[T, Sequence[T]]) -> List[T]:
    if isinstance(val, tuple):
        return list(val)
    if isinstance(val, list):
        return val
    return [val]  # type: ignore


def prod(vals: Sequence[int]) -> int:
    return reduce(lambda x, y: x * y, vals)


def closest_power_2(x: float) -> int:
    exponent = log2(x)
    distance_fn = lambda z: abs(x - 2 ** z)  # noqa
    exponent_closest = min((floor(exponent), ceil(exponent)), key=distance_fn)
    return 2 ** int(exponent_closest)

def rand_bool(shape, proba, device = None):
    if proba == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif proba == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.bernoulli(torch.full(shape, proba, device=device)).to(torch.bool)


"""
Kwargs Utils
"""


def group_dict_by_prefix(prefix: str, d: Dict) -> Tuple[Dict, Dict]:
    return_dicts: Tuple[Dict, Dict] = ({}, {})
    for key in d.keys():
        no_prefix = int(not key.startswith(prefix))
        return_dicts[no_prefix][key] = d[key]
    return return_dicts


def groupby(prefix: str, d: Dict, keep_prefix: bool = False) -> Tuple[Dict, Dict]:
    kwargs_with_prefix, kwargs = group_dict_by_prefix(prefix, d)
    if keep_prefix:
        return kwargs_with_prefix, kwargs
    kwargs_no_prefix = {k[len(prefix) :]: v for k, v in kwargs_with_prefix.items()}
    return kwargs_no_prefix, kwargs


def prefix_dict(prefix: str, d: Dict) -> Dict:
    return {prefix + str(k): v for k, v in d.items()}


def log_print(message, logger):
    logger.info(message)
    print(message)


def merge_batches(batch1, batch2):
    batch1 = [b.squeeze() for b in batch1]
    batch2 = [b.squeeze() for b in batch2]

    waves, texts1, input_lengths1, mels1, mel_input_length1, adj_texts1, ref_lengths1, adj_mels1, adj_mels_lengths1 = batch1
    waves2, texts2, input_lengths2, mels2, mel_input_length2, adj_texts2, ref_lengths2, adj_mels2, adj_mels_lengths2 = batch2
    
    waves = [w for w in waves]
    waves2 = [w for w in waves2]
    waves.extend(waves2)
    
    nmels = mels1[0].size(0)
    max_text_length = max(input_lengths1.max(), input_lengths2.max())
    max_mel_length = max(mel_input_length1.max(), mel_input_length2.max())

    max_adjmel_length = max(adj_mels_lengths1.max(), adj_mels_lengths2.max())
    max_ref_length = max(ref_lengths1.max(), ref_lengths2.max())
    
    batch_size = mels1.size(0) + mels2.size(0)
    
    mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
    texts = torch.zeros((batch_size, max_text_length)).long()
    input_lengths = torch.zeros(batch_size).long()
    mel_input_length = torch.zeros(batch_size).long()
    adj_mels = torch.zeros((batch_size, nmels, max_adjmel_length)).float()
    adj_mels_lengths = torch.zeros(batch_size).long()
    ref_texts = torch.zeros((batch_size, max_ref_length)).long()
    ref_lengths = torch.zeros(batch_size).long()
    
    mels[:mels1.size(0), :, :mels1.size(-1)] = mels1
    mels[mels1.size(0):, :, :mels2.size(-1)] = mels2

    texts[:mels1.size(0), :texts1.size(-1)] = texts1
    texts[mels1.size(0):, :texts2.size(-1)] = texts2

    input_lengths[:mels1.size(0)] = input_lengths1
    input_lengths[mels1.size(0):] = input_lengths2

    mel_input_length[:mels1.size(0)] = mel_input_length1
    mel_input_length[mels1.size(0):] = mel_input_length2

    adj_mels[:adj_mels1.size(0), :, :adj_mels1.size(-1)] = adj_mels1
    adj_mels[adj_mels1.size(0):, :, :adj_mels2.size(-1)] = adj_mels2

    adj_mels_lengths[:mels1.size(0)] = adj_mels_lengths1
    adj_mels_lengths[mels1.size(0):] = adj_mels_lengths2

    ref_texts[:mels1.size(0), :adj_texts1.size(-1)] = adj_texts1
    ref_texts[mels1.size(0):, :adj_texts2.size(-1)] = adj_texts2

    ref_lengths[:mels1.size(0)] = ref_lengths1
    ref_lengths[mels1.size(0):] = ref_lengths2

    return waves, texts, input_lengths, mels, mel_input_length, ref_texts, ref_lengths, adj_mels, adj_mels_lengths


def random_mask_tokens(input_tensor, M, part=5):
    """
    Randomly mask tokens in the input tensor, ensuring at least M portion remains unmasked.

    Args:
    input_tensor (torch.Tensor): The input tensor of shape [512, T].
    M (float): The minimum portion of tokens that should remain unmasked.

    Returns:
    torch.Tensor: The masked input tensor.
    """
    B, T = input_tensor.shape

    if T <= M:
        return input_tensor
    masked_part = np.random.randint(0, part)
    
    max_mask = T - M
    masked_len = 0
    
    masked_tensor = input_tensor.clone()
    for i in range(masked_part):
        mask_start = np.random.randint(0, T)
        mask_end = np.random.randint(mask_start, T)
        
        if (mask_end - mask_start) + masked_len > max_mask:
            continue
            
        masked_tensor[:, mask_start:mask_end] = 0
        
        masked_len += (mask_end - mask_start)

    return masked_tensor

def to_batch(
    batch_size: int,
    device: torch.device,
    x: Optional[float] = None,
    xs: Optional[Tensor] = None,
) -> Tensor:
    assert exists(x) ^ exists(xs), "Either x or xs must be provided"
    # If x provided use the same for all batch items
    if exists(x):
        xs = torch.full(size=(batch_size,), fill_value=x).to(device)
    assert exists(xs)
    return xs