def replace_special_cases(text):
    replacements = {
        "пк": "гж",
        "К)": "Ю",
        "к)": "ю",
        ")к": "ж",
        "2к": "ж",
        "*с": "ж",
        "х(": "ж",
        "/ю" : "лс"
    }
    
    def replace_word(word):
        for key in replacements:
            if key in word:
                word = word.replace(key, replacements[key])
        return word

    lines = text.split("\n")
    return "\n".join(" ".join(replace_word(word) for word in line.split()) for line in lines)

def replace_p(text):
    mongolian_words_with_p = {
        "поп", "паспорт", "пицца", "паб", "принтер", "программа",
        "проблема", "процесс", "паар", "памперс", "паян", "паахай",
        "пил", "пийшин"
    }
    
    lines = text.split("\n")
    
    def replace_word(word):
        if "П" in word and word.lower() not in mongolian_words_with_p:
            return word.replace("П", "Л")
        return word

    return "\n".join(" ".join(replace_word(word) for word in line.split()) for line in lines)

def fix_uppercase_yo(text):
    lines = text.split('\n')
    result = []
    for line in lines:
        words = line.split(' ')
        fixed_words = []
        for w in words:
            if 'Ё' in w and w.replace('Ё', '').islower():
                w = w.replace('Ё', 'ё')
            fixed_words.append(w)
        result.append(' '.join(fixed_words))
    return '\n'.join(result)

def remove_noise(text):
    def is_mongolian(ch):
        mn_lower = 'авгдеёжзийклмнопрстуфхцчшщъыьэюяөү'
        return ch.lower() in mn_lower
    lines = text.split('\n')
    result = []
    for line in lines:
        words = line.split(' ')
        fixed_words = []
        for w in words:
            chars = list(w)
            i = 1
            while i < len(chars) - 1:
                if chars[i] in ['.', ':', ';'] and is_mongolian(chars[i - 1]) and is_mongolian(chars[i + 1]):
                    chars.pop(i)
                else:
                    i += 1
            w = ''.join(chars)
            fixed_words.append(w)
        result.append(' '.join(fixed_words))
    return '\n'.join(result)

def convert_to_lower(text):
    def is_valid(s):
        return len(s) > 2 and sum(c.isupper() for c in s[1:]) >= 1 and s[0].islower() and sum(c.islower() for c in s[1:]) >= 1

    lines = text.split('\n')
    result = []
    for line in lines:
        words = line.split(' ')
        fixed_words = []
        for w in words:
            if is_valid(w):
                w = w.lower()
            fixed_words.append(w)
        result.append(' '.join(fixed_words))
    return '\n'.join(result)

def convert_digit_to_letter(text):
    mn_lower = 'авгдеёжзийклмнопрстуфхцчшщъыьэюяөү'
    lines = text.split('\n')
    result = []
    for line in lines:
        words = line.split(' ')
        new_words = []
        for word in words:
            chars = list(word)
            if len(chars)>2:
                for i in range(1, len(chars)-1):
                    if chars[i].isdigit() and chars[i-1].lower() in mn_lower and chars[i+1].lower() in mn_lower:
                        if chars[i] == '0':
                            chars[i] = 'о'
                        elif chars[i] == '5':
                            chars[i] = 'э'
            new_words.append(''.join(chars))
        result.append(' '.join(new_words))
    return '\n'.join(result)

def post_process(result_full_text):

    processed = replace_special_cases(result_full_text)
    processed = replace_p(processed)
    processed = fix_uppercase_yo(processed)
    processed = remove_noise(processed)
    processed = convert_to_lower(processed)
    processed = convert_digit_to_letter(processed)
    return processed