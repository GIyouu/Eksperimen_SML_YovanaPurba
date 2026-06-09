import pandas as pd

def preprocess_data(input_file, output_file):

    # Load data
    df = pd.read_csv(input_file)

    # Drop kolom yang tidak digunakan
    df = df.drop(
        columns=[
            'PassengerId',
            'Name',
            'Ticket',
            'Cabin'
        ]
    )

    # Handle missing value
    df['Age'] = df['Age'].fillna(
        df['Age'].median()
    )

    df['Embarked'] = df['Embarked'].fillna(
        df['Embarked'].mode()[0]
    )

    # Encoding
    df['Sex'] = df['Sex'].map({
        'male': 0,
        'female': 1
    })

    df = pd.get_dummies(
        df,
        columns=['Embarked'],
        drop_first=True
    )

    # Save hasil preprocessing
    df.to_csv(
        output_file,
        index=False
    )

    print(f"Preprocessed data saved to {output_file}")


if __name__ == "__main__":
    preprocess_data(
        "../titanic_raw/Titanic-Dataset.csv",
        "titanic_preprocessing.csv"
    )