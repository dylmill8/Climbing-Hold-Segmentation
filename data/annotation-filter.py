import pandas as pd

# filters out empty annotations for uploading pre-labeled data to Roboflow
def filter_annotations(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    filtered = df[df['region_shape_attributes'].notna() & (df['region_shape_attributes'] != '{}')]

    filtered.to_csv(output_csv, index=False)

def main():
    filter_annotations(".\\data\\bh\\bh-annotation.csv", ".\\data\\bh\\bh-annotation_filtered.csv")
    filter_annotations(".\\data\\bh-phone\\bh-phone-annotation.csv", ".\\data\\bh-phone\\bh-phone-annotation_filtered.csv")
    filter_annotations(".\\data\\sm\\sm-annotation.csv", ".\\data\\sm\\sm-annotation_filtered.csv")

if __name__ == "__main__":
    main()