import os
import pandas as pd


# Core Hindi-Mundari Dictionary & Pronoun/Noun Lexicon Entries
LEXICON_PAIRS = [
    # Pronouns & Possessives
    ("तुम्हारा", "अमअः"),
    ("मेरा", "अञअः"),
    ("उसका", "इनिअः"),
    ("हमारा", "आलेअः"),
    ("उनका", "इनकुअः"),
    ("मैं", "अइङ"),
    ("तुम", "अम"),
    ("वह", "इनिः"),
    ("हम", "आबु"),
    ("वे", "इनकु"),
    ("मुझे", "अञके"),
    ("तुम्हें", "अमके"),
    ("उसे", "इनिःके"),
    
    # Nouns
    ("घर", "ओड़ाः"),
    ("नाम", "नुतुम"),
    ("पानी", "दाः"),
    ("खाना", "मंडी"),
    ("गाँव", "हातु"),
    ("आदमी", "होड़ो"),
    ("लड़का", "कोड़ा"),
    ("लड़की", "कुड़ि"),
    ("माता", "एंगा"),
    ("पिता", "अपु"),
    ("भाई", "हागा"),
    ("बहन", "मिसि"),
    ("पेड़", "दारु"),
    ("फल", "जो"),
    ("फूल", "बा"),
    ("दिन", "मुसिङ"),
    ("रात", "निदा"),
    ("सूरज", "सिंगी"),
    ("चाँद", "चानडुः"),
    ("हवा", "होयो"),
    ("ज़मीन", "ओते"),
    ("आकाश", "सिरमा"),

    # Verbs & Common Expressions
    ("जाना", "सेनोः"),
    ("आना", "हिजुः"),
    ("खाना", "जोम"),
    ("पीना", "नू"),
    ("देखना", "लेल"),
    ("सुनना", "अयुम"),
    ("बोलना", "काजी"),
    ("करना", "कामी"),
    ("सोना", "गिदिः"),
    ("उठना", "बिरिद"),
    ("देना", "ओम"),
    ("लेना", "इदि"),
    ("रहना", "ताइन"),
    
    # Common Questions & Greetings
    ("क्या", "चिनाः"),
    ("कहाँ", "ओकेरे"),
    ("कौन", "ओकोए"),
    ("क्यों", "चिकाते"),
    ("कब", "चिमता"),
    ("कैसे", "चिलका"),
    ("नमस्कार", "जोहाार"),
    ("धन्यवाद", "सराहना"),
]

TRAIN_CSV = "data/processed/train/train.csv"


def augment_training_data(repetition_factor=10):
    print("--- Dictionary Augmentation for Training Set ---")

    if not os.path.exists(TRAIN_CSV):
        print(f"Error: {TRAIN_CSV} not found!")
        return

    existing_df = pd.read_csv(TRAIN_CSV)
    print(f"Original Training Pairs: {len(existing_df)}")

    # Create DataFrame from Lexicon Pairs
    lexicon_records = []
    for hindi, mundari in LEXICON_PAIRS:
        lexicon_records.append({
            "Hindi": hindi,
            "Mundari": mundari,
            "source": "Lexicon_Dictionary"
        })

    lexicon_df = pd.DataFrame(lexicon_records)

    # Repeat lexicon pairs so the model learns isolated word associations alongside long sentences
    repeated_lexicon_df = pd.concat([lexicon_df] * repetition_factor, ignore_index=True)

    updated_df = pd.concat([existing_df, repeated_lexicon_df], ignore_index=True)

    updated_df.to_csv(TRAIN_CSV, index=False)
    print(f"Added {len(repeated_lexicon_df)} dictionary pairs ({len(LEXICON_PAIRS)} unique words repeated {repetition_factor}x).")
    print(f"New Training Set Total Size: {len(updated_df)} pairs")
    print(f"Updated dataset saved to: {TRAIN_CSV}")


if __name__ == "__main__":
    augment_training_data(repetition_factor=20)
