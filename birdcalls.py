import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
import librosa
from librosa.core import load
from soundfile import write
from scipy.io import wavfile
import os
import scipy.misc


class Crawler:
    '''
    A crawler class for the bird sounds API provided by
    https://www.xeno-canto.org
    Processes a number of queries to the API, dowloads respective files and
    does some postprocessing.
    Attributes:
        verbose (bool): whether to be verbose
        api_endpoint (str): defaults to the endpoint of xeno-canto
        queries (list): list of dicts specifying API queries. See api_query() for details
        df (pandas df): df to save api responses
        root (string): path to root where data is to be stored
        save_img (bool): whether to create images from the spectogram arrays
        make_wav (bool): whether to convert sounds to .wav
        make_mel_spec (bool): whether to make mel-spectrograms from the sounds
        sr (int): sampling rate in Hz
        n_mels (int): number of channels for mel-spectrograms
        hop_length (int): number of consecutive measures combined into one bin
                            for mel-spectrograms
        extract_chunks (bool): whether to extract and save fix sized chunks from spectrograms
        n_chuncks (int): number of chucks to extract from one spectogram
        len_chuncks (int):
    Note:
        mel-spectrograms are saved as ndarrays with shape (n_mels, #sec*sr/hop_length).
        The dic format for a query as taken by api_query() must be:
        {'genus':<.>, 'country':<.>, 'quality':<.>,
                    'area':<.>, 'page':<.>}
        quality can be from {'q:A', 'q>:B' , 'q<:C'}
    '''

    def __init__(self, queries=None, root=None):
        '''
        initializes default attributes.
        '''
        # Todo: implement info dataframe
        self.queries = queries
        self.verbose = True
        # API requests
        self.api_endpoint = 'http://www.xeno-canto.org/api/2/recordings'
        self.df = None
        # saving the data
        self.root = None
        self.save_img = True
        self.info_df = None
        self.min_length = .96 # in seconds, not in use
        self.convert_to_wav = True
        self.keep_mp3 = False
        self.max = None
        # processing the data
        self.make_mel_spec = True
        self.sr = 22050
        self.input_sr = 16000
        self.n_mels = 64
        self.hop_length = 256
        # extracting chuncks
        self.extract_chunks = True
        self.n_chunks = 10
        self.len_chunks = 96 # in bins of mel spec

    def api_query(self, query):
        '''
        returns a query url as expected by the Xeno-Canto API from a query dictionary
        Args:
            query (dic): in the format {'genus':<.>, 'country':<.>, 'quality':<.>,
                        'area':<.>, 'page':<.>}
        '''
        string = self.api_endpoint + '?query='

        genus = query['genus'] if 'genus' in query else None
        area = query['area'] if 'area' in query else None
        country = query['country'] if 'country' in query else None
        quality = query['quality'] if 'quality' in query else None
        page = query['page'] if 'page' in query else None

        #string=API_ENDPOINT + '?query='

        if((page is None) or (page == 0) or (page == '')):
            page = 1

        if genus is None:
            None
        else:
            string = string + 'gen:'+ str(genus)

        if area is None:
            None
        else:
            string = string + '%20' + 'area:' + str(area)

        if country is None:
            None
        else:
            string = string + '%20' + 'cnt:' + str(country)

        if quality is None:
            None
        else:
            string = string + '%20' + str(quality)

        string = string + '&page=' + str(page)

        return string


    def query(self, queries=None):
        '''
        applies all queries in self.queries and stores responses in self.df
        '''
        if queries is not None:
            self.queries = queries
        if self.queries is None:
            raise Exception('No queries have been instatiated, yet.')
        frames = []
        for i, q in enumerate(self.queries):
            query = self.api_query(q)
            if self.verbose:
                print('processing query: {}'.format(query))
            result = requests.get(query).json()['recordings']
            df_temp = pd.DataFrame(result)
            frames.append(df_temp)
        if self.verbose:
            print('updating self.df with recent query responses.')
        self.df = pd.concat(frames, ignore_index=True)


    def get_summary(self):
        '''
        returns a dataframe including all unique genii, their number of instances.
        '''
        #TODO: report total recording time
        summary = pd.DataFrame({'labels': self.df['gen'].unique()})
        summary['counts'] = [self.df['gen'].where(self.df['gen']==gen).count() \
                            for gen in summary['labels']]
        return summary


    def download(self, save_to=None):
        '''
        downloads all files in API response and generates a a directory tree with
        structure save_to/{gen}/{instances}.
        If self.convert_to_wav is true a .wav file is generated.
        If self.keep_mp3 is false resp mp3 is deleted.
        PREFER this function and use Dataset.process_xeno_canto_downloads() to
        process downloads further.
        '''
        if save_to is not None:
            self.root = save_to
        if self.df is None:
            raise Exception('No query has been applied to the API, yet. Run query() first')
        labels = self.df.gen.unique()
        if self.max is not None:
            counts = {i:0 for i in labels}
        print('downloading a maximum of {} instances/class'.format(self.max))
        for label in list(self.df.gen.unique()):
            new_dir = os.path.join(self.root, label)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
        for index, row in self.df.iterrows():
            label = str(row.gen)
            if self.max is not None:
                if counts[label] < self.max:
                    counts[label] += 1
                else:
                    continue
            url = 'http:' + row.file
            path = os.path.join(self.root,label)
            mp3_file = os.path.join(path, '{}.mp3'.format(str(row.id)))
            wav_file = os.path.join(path, '{}.wav'.format(str(row.id)))

            if os.path.exists(wav_file) or os.path.exists(mp3_file):
                print(f"skipping {url}. file already downloaded.")
            else:
                print(f"downloading {url} to {mp3_file}")
                r = requests.get(url, timeout=60)
                open(mp3_file, 'wb').write(r.content)

                if self.convert_to_wav:
                    y, sr = load(mp3_file)
                    y *= 32768
                    y = y.astype(np.int16)
                    wavfile.write(wav_file, rate=22050, data=y)
                if not self.keep_mp3:
                    os.remove(mp3_file)


    def process_sounds(self):
        '''
        processes dowloaded files below self.root after running download_files().
        DEPRECATED Don't use this for the pretrained VGGish!
        TODO: this should go to preprocessing if kept at all
        '''
        self.info_df = self.df[['gen', 'id']].copy()
        for path, dirs, files in os.walk(self.root):
            for file in files:
                if file.endswith('.mp3'):
                    y, sr = load(os.path.join(path, file))
                    if self.convert_to_wav:
                        write(os.path.join(path, file.replace('.mp3', '.wav')), y, self.input_sr)
                    if self.make_mel_spec:
                        S = librosa.feature.melspectrogram(y, sr=self.sr, n_mels=self.n_mels,
                                                    hop_length=self.hop_length)
                        log_S = librosa.amplitude_to_db(S, ref=np.max)
                        np.save(os.path.join(path, 'mel_spec.npy'), log_S)
                        if self.save_img:
                            scipy.misc.imsave(os.path.join(path, 'mel_spec.jpg'), log_S)
                        if self.extract_chunks:
                            if log_S.shape[1] < self.len_chunks:
                                print('recording {} has length {} which is shorter \
                                        than required chunk length.')
                                continue
                            self.spec_chunks(log_S, path=path)
        self.info_df.to_csv(os.path.join(self.root, 'info.csv'), sep='\t')


    def remove_processed(self):
        '''
        removes all .wav, .jpg and .npy files under self.root
        '''
        for path, dirs, files in os.walk(self.root):
            for file in files:
                if file.endswith('.wav') \
                    or file.endswith('.npy') \
                    or file.endswith('.jpg'):
                    os.remove(os.path.join(path, file))


if __name__ == '__main__':

    queries = [{'genus':'Parus', 'quality':'q_gt:B'},
                 {'genus':'Turdus', 'quality':'q_gt:B'},
                 {'genus':'Garrulus', 'quality':'q_gt:B'}]

    crawler = Crawler(queries)
    crawler.query()
    summary = crawler.get_summary()
    print(summary)
    crawler.download(save_to='data/10birds')