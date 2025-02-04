import re
from collections import Counter
# IPA Phonemizer: https://github.com/bootphon/phonemizer

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'-"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i

class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
        print(len(dicts))
    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                pass
                print(text)
                print(char)
        return indexes


def check_unused_tokens(dataset_path):
    """
    Check unused tokens in the dataset.
    Args:
        dataset_path (str): Path to dataset file (format: filename|text|id)
    Returns:
        list: List of unused tokens
    """
    # Collect all characters from dataset
    used_chars = set()
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Split by | and take text (2nd element)
            text = line.strip().split('|')[1]
            used_chars.update(set(text))
    
    # Compare with defined tokens
    unused_tokens = []
    for token in symbols:
        if token not in used_chars:
            unused_tokens.append(token)
    
    # Display statistics
    print(f"Total number of defined tokens: {len(symbols)}")
    print(f"Number of used tokens: {len(used_chars)}")
    print(f"Number of unused tokens: {len(unused_tokens)}")
    
    return unused_tokens

def phonem_before_this_phoneme(phoneme, datasetpath):
    """
    Check phonemes that appear before a given phoneme in the dataset.
    Args:
        phoneme (str): Phoneme to search for
        datasetpath (str): Path to dataset file (format: filename|text|id)
    Returns:
        list: List of phonemes that appear before the given phoneme
    """
    phoneme_before = set()
    
    with open(datasetpath, 'r', encoding='utf-8') as f:
        for line in f:
            # Split by | and take text (2nd element)
            text = line.strip().split('|')[2]
            # Seek the phoneme in the text
            if phoneme in text:
                # Get the index of the phoneme
                index = text.index(phoneme)
                # Get the phoneme before the phoneme
                phoneme_before.add(text[index-1])
    
    return phoneme_before


def extract_parentheses_content(dataset_path):
    """
    Extract and count all texts between parentheses in the dataset.
    Args:
        dataset_path (str): Path to dataset file (format: filename|text|id)
    Returns:
        dict: Dictionary of contents between parentheses and their occurrence count
    """
    parentheses_content = []
    pattern = r'\((.*?)\)'
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip().split('|')[2]
            # Find all contents between parentheses
            matches = re.findall(pattern, text)
            parentheses_content.extend(matches)
    
    # Count occurrences
    content_counter = Counter(parentheses_content)
    
    # Display statistics
    print(f"Total number of elements between parentheses: {len(parentheses_content)}")
    print(f"Number of unique elements: {len(content_counter)}")
    
    return dict(content_counter)

def clean_parentheses(input_path, output_path):
    """
    Clean parentheses and their content from the dataset.
    """
    pattern = r'\([^)]*\)'  # Pattern to find (text)
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            # Split line into components
            parts = line.strip().split('|')
            
            # Clean text (middle part)
            cleaned_text = re.sub(pattern, '', parts[2])
            # Remove potential double spaces
            cleaned_text = ' '.join(cleaned_text.split())
            
            # Reassemble the line
            new_line = f"{parts[0]}|{cleaned_text}|{parts[3]}\n"
            f_out.write(new_line)

def replace_phoneme(input_path, output_path, replacements):
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            # Split line into components
            parts = line.strip().split('|')
            
            # Replace specific phonemes
            for phoneme, replacement in replacements.items():
                parts[2] = parts[2].replace(phoneme, replacement)
            
            # Reassemble the line
            new_line = f"{parts[0]}||{parts[2]}|{parts[3]}|\n"
            f_out.write(new_line)



if __name__ == "__main__":
    unused = check_unused_tokens("Data/train_list_original_mp3.txt")
    print("Unused tokens:", unused)
    before_tild = phonem_before_this_phoneme("̃", "Data/train_list_original_mp3.txt")
    print("Phonemes before ̃:", before_tild)
    before_minus = phonem_before_this_phoneme("-", "Data/train_list_original_mp3.txt")
    print("Phonemes before -:", before_minus)
    selected = ['ʘ', 'ɺ', 'ɻ', 'ʀ','ǂ']
    selected_for_minus = ['ʧ', 'ʉ', 'ʋ', 'ⱱ']

    transphorm = {}

    for phoneme in before_tild:
        print(phoneme)
        # Take one of the selected phonemes
        transphorm[phoneme+"̃"] = selected.pop(0)

    for phoneme in before_minus:
        transphorm[phoneme+"-"] = selected_for_minus.pop(0)
    print(transphorm)
    #replace_phoneme("Data/train_list.txt","Data/train_list_replaced.txt",transphorm)
    #replace_phoneme("Data/val_list.txt","Data/val_list_replaced.txt",transphorm)