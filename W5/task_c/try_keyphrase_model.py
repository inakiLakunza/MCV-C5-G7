
from keyphrasetransformer import KeyPhraseTransformer


if __name__ == '__main__':

    kp = KeyPhraseTransformer()

    txt_list = [
        'A black cat is inside a white toilet.',                            # 589
        'A cute kitten is sitting in a dish on a table.',                   # 754
        'A brown and black horse in the middle of the city eating grass.',  # 961
        'Dogs stick their heads out of car windows',                        # 1113
        'A cat in between two cars in a parking lot.',                      # 1536
        'A man sits next to his horse in an alley in costume.',             # 6101
        'Two horses eating grass near a parking lot. '                      # 6853
    ]
    
    for txt in txt_list:

        topics = kp.get_key_phrases(txt)

        print(f"Analyzed text: \n{txt}\n")
        print(f"Extracted topics:")

        for topic in topics: print(topic)

        print("\n\n\n --------------------- \n")