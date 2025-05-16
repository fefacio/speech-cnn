import sys
import pandas as pd
import torch
from torchvision import models
import os

from path_constants import *

def main():
    # existing_files = [file.split('.')[0] for file in os.listdir(RECORDINGS_PATH)]
    # print(f'ef: {existing_files}')
    # for argument in sys.argv:
    #     print(argument)

    # df = pd.read_csv("./speakers_all.csv")


    
    # usa_df = df[df['country'] == 'usa']
    # non_usa_df = df[df['country'] != 'usa']

    # # Reduz pela metade os casos dos EUA
    # usa_sampled = usa_df.sample(frac=0.2, random_state=42)
    # print(len(usa_sampled))
    # # Junta com o resto
    # df_balanceado = pd.concat([non_usa_df, usa_sampled], ignore_index=True)

    # # (opcional) Embaralhar
    # df_balanceado = df_balanceado.sample(frac=1, random_state=42).reset_index(drop=True)

    # # allowed_countries = {'usa', 'canada', 'uk', 'australia'}
    # # df = df[
    # #     (df['country'].str.lower().isin(allowed_countries)) &
    # #     (df['native_language'].str.lower() == 'english')
    # # ]
    # row = df.iloc[500]
    # classes = {"usa": 0, "canada": 1, "uk": 2, "australia": 3}
    # num_classes = len(classes)
    # label_idx = classes[row['country']]
    # print(f'total {len(df[(df['country']=='usa') & (df['native_language']=='english')])}')
    # print(f'total {len(df[(df['country']=='usa') & ~(df['native_language']=='english')])}')
    # print(label_idx)
    # print(torch.nn.functional.one_hot(torch.tensor(label_idx), num_classes))
    # print(df['country'].value_counts(dropna=False))
    print('vgg16 summary')
    print(models.vgg16(weights=models.VGG16_Weights.DEFAULT))

if __name__ == "__main__":
    main()