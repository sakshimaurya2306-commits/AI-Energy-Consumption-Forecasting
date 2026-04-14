from src.data_loader import load_data
from src.features import create_features
from src.train import train_models

def main():

    df = load_data("data/energy.csv")
    df = create_features(df)

    # ✅ ADD THIS LINE HERE
    print("Data shape after features:", df.shape)

    X = df.drop(columns=['Energy'])
    y = df['Energy']

    train_models(X, y)

if __name__ == "__main__":
    main()