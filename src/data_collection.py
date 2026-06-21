import pandas as pd


def load_isot():
    isot_true = pd.read_csv("data/raw/True.csv")
    isot_fake = pd.read_csv("data/raw/Fake.csv")

    isot_true["label"] = 1
    isot_fake["label"] = 0

    isot_data = pd.concat([isot_true, isot_fake], axis=0)
    isot_data = isot_data[["text", "label"]]

    print(f"ISOT samples: {len(isot_data)}")

    return isot_data


def load_liar():
    liar_cols = [
        "id", "label_raw", "statement", "subject", "speaker",
        "job", "state", "party", "barely_true", "false",
        "half_true", "mostly_true", "pants_on_fire", "context"
    ]

    l_train = pd.read_csv("data/raw/train.tsv", sep="\t", names=liar_cols)
    l_test = pd.read_csv("data/raw/test.tsv", sep="\t", names=liar_cols)
    l_valid = pd.read_csv("data/raw/valid.tsv", sep="\t", names=liar_cols)

    liar_data = pd.concat([l_train, l_test, l_valid], axis=0)

    label_map = {
        "true": 1,
        "mostly-true": 1,
        "half-true": 1,
        "false": 0,
        "barely-true": 0,
        "pants-fire": 0,
    }

    liar_data["label"] = liar_data["label_raw"].map(label_map)

    liar_data = liar_data[["statement", "label"]]
    liar_data = liar_data.rename(columns={"statement": "text"})

    print(f"LIAR samples: {len(liar_data)}")

    return liar_data


def load_welfake():
    welfake = pd.read_csv("data/raw/WELFake_Dataset.csv")
    welfake = welfake[["text", "label"]]

    print(f"WELFake samples: {len(welfake)}")

    return welfake


def load_all_data():
    isot = load_isot()
    liar = load_liar()
    welfake = load_welfake()

    final_df = pd.concat([isot, liar, welfake], ignore_index=True)

    print(f"Total samples: {len(final_df)}")

    return final_df


if __name__ == "__main__":
    data = load_all_data()
