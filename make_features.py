import os
import librosa
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def get_audios(folderpath):
    """Происходит чтение звуковых файлов из заданного пути folderpath"""

    fullpaths, audios = np.empty(0), np.empty(0)

    for every_folder in os.listdir(folderpath):
        for every_file in os.listdir(folderpath + every_folder):
            if every_file == '.DS_Store':
                continue
            fullpaths = np.append(fullpaths, str(folderpath) + str(every_folder) + '/' + str(every_file))

    audios = [np.append(audios, librosa.load(path)[0]) for path in fullpaths]
    paths_for_df = [(path.split('/')[-2] +'/'+ path.split('/')[-1]) for path in fullpaths]
    return paths_for_df, audios


def make_targets(paths_for_df):
    """Строим разметку данных (= целевые метки). Если в пути есть ключевое слово - ставим соответствующее число"""

    targets = np.empty(0)
    for path in paths_for_df:
        type_of_noise = path.split('/')[0]
        if type_of_noise == 'construction':
            targets = np.append(targets, '1')
        elif type_of_noise == 'people':
            targets = np.append(targets, '2')
        elif type_of_noise == 'railway':
            targets = np.append(targets, '3')
        elif type_of_noise == 'roads':
            targets = np.append(targets, '4')
        else:
            targets = np.append(targets, '0')
    return targets


def get_base_features(audios):
    """Получаем базовые признаки звуковых файлов"""

    mfccs, zero_crossing_rate = np.empty(0), np.empty(0)
    mean, std, var = np.empty(0), np.empty(0), np.empty(0) 
    min_, max_ = np.empty(0), np.empty(0)

    for y in audios:
        """[Mean] Mel-frequency cepstral coefficients"""
        mfccs = np.append(mfccs, np.mean(librosa.feature.mfcc(y=y)))
        mean = np.append(mean, np.mean(y))
        std = np.append(std, np.std(y))
        var = np.append(var, np.var(y))
        min_ = np.append(min_, np.min(y))
        max_ = np.append(max_, np.max(y))
        zero_crossing_rate = np.append(zero_crossing_rate, np.mean(librosa.feature.zero_crossing_rate(y=y)))

    return mfccs, mean, std, var, min_, max_, zero_crossing_rate


def make_base_raw_data(paths_for_df, targets, mfccs, mean, std, var, min_, max_, zero_crossing_rate):
    """Формируем датафрейм с базовыми признаками"""

    df_base = pd.DataFrame({'Paths':paths_for_df,
                       'Type_of_noise':targets, 
                       'MFCCs (mean)':mfccs, 
                       'Mean':mean,
                       'Std':std,
                       'Var':var,
                       'Min':min_,
                       'Max':max_,
                       'Zero crossing (mean)':zero_crossing_rate})
    df_base.to_csv('noises_features.csv', sep=',')
    return df_base


def get_additional_features(audios):
    """Получаем дополнительные признаки звуковых файлов"""

    chroma_stft, chroma_cqt, chroma_cens = np.empty(0), np.empty(0), np.empty(0)
    melspectrogram, rmse, tonnetz = np.empty(0), np.empty(0), np.empty(0)
    spectral_centroid = np.empty(0)
    spectral_bandwidth = np.empty(0)
    spectral_contrast = np.empty(0)
    spectral_rolloff = np.empty(0)

    for y in audios:
        """Compute a chromagram from a waveform or power spectrogram."""
        chroma_stft = np.append(chroma_stft, np.mean(librosa.feature.chroma_stft(y=y)))

        """Constant-Q chromagram"""
        chroma_cqt = np.append(chroma_cqt, np.mean(librosa.feature.chroma_cqt(y=y)))

        """Computes the chroma variant “Chroma Energy Normalized” (CENS)"""
        chroma_cens = np.append(chroma_cens, np.mean(librosa.feature.chroma_cens(y=y)))

        """Compute a mel-scaled spectrogram"""
        melspectrogram = np.append(melspectrogram, np.mean(librosa.feature.melspectrogram(y=y)))

        """Compute root-mean-square (RMS) energy for each frame, either from the audio samples y or from a spectrogram S."""
        rmse = np.append(rmse, np.mean(librosa.feature.rmse(y=y)))

        """Compute the spectral centroid."""
        spectral_centroid = np.append(spectral_centroid, np.mean(librosa.feature.spectral_centroid(y=y)))

        """Compute p’th-order spectral bandwidth (default p=2)"""
        spectral_bandwidth = np.append(spectral_bandwidth, np.mean(librosa.feature.spectral_bandwidth(y=y)))

        """Compute spectral contrast"""
        spectral_contrast = np.append(spectral_contrast, np.mean(librosa.feature.spectral_contrast(y=y)))

        """Compute roll-off frequency"""
        spectral_rolloff = np.append(spectral_rolloff, np.mean(librosa.feature.spectral_rolloff(y=y)))

        """Computes the tonal centroid features (tonnetz)"""
        tonnetz = np.append(tonnetz, np.mean(librosa.feature.tonnetz(y=y)))

    return chroma_stft, chroma_cens, chroma_cqt, melspectrogram, rmse, spectral_centroid, spectral_bandwidth, spectral_contrast, spectral_rolloff, tonnetz


def make_additional_raw_data(paths_for_df, chroma_stft, chroma_cens, chroma_cqt, melspectrogram, rmse, spectral_centroid, spectral_bandwidth, spectral_contrast, spectral_rolloff, tonnetz):
    """Формируем датафрейм с дополнительными признаками"""

    df_additional = pd.DataFrame({'Paths':paths_for_df, 
                       'Chroma stft (mean)':chroma_stft, 
                       'Chroma cens (mean)':chroma_cens,
                       'Chroma cqt (mean)':chroma_cqt,
                       'Melspectrogram (mean)':melspectrogram,
                       'RMSE (mean)':rmse,
                       'Spectral centroid (mean)':spectral_centroid,
                       'Spectral bandwidth (mean)':spectral_bandwidth,
                       'Spectral contrast (mean)':spectral_contrast,
                       'Spectral rolloff (mean)':spectral_rolloff,
                       'Tonnetz (mean)':tonnetz})
    df_additional.to_csv('noises_additional_features.csv', sep=',')
    return df_additional


def merge_features():
    """Соединяем датафреймы по стобцу путей, чтобы получить датафрейм и с базовыми, и с дополнительными признаками
    звуковых файлов"""

    df_base = pd.read_csv('noises_features.csv', sep=',', index_col=0)
    df_additional = pd.read_csv('noises_additional_features.csv', sep=',', index_col=0)
    df_result = pd.merge(df_base, df_additional, on='Paths')
    df_result.to_csv('full_features.csv', sep=',', index=False)
    return df_result



def main():
    folderpath = '/Users/User/Desktop/noises/'
    audios = get_audios(folderpath)
    targets = make_targets(audios[0])
    base_features = get_base_features(audios[1])
    base_raw_data = make_base_raw_data(audios[0], targets, *base_features)
    additional_features = get_additional_features(audios[1])
    additional_raw_data = make_additional_raw_data(audios[0], *additional_features)
    features_merging = merge_features()



if __name__ == '__main__':
    main()