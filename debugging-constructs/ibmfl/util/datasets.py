"""
Licensed Materials - Property of IBM
Restricted Materials of IBM
20221069
© Copyright IBM Corp. 2022 All Rights Reserved.
"""
"""
Module providing utility functions for loading datasets for use in FL
"""
import os
import sys
import shutil
from zipfile import ZipFile
import json
import numpy as np
import pandas as pd
import requests
import gzip
import gensim.downloader as api
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess


def save_file(path, url):
    """
    Saves a file from URL to path

    :param path: the path to save the file
    :type path; `str`
    :param url: the link to download from
    :type url: `str`
    """
    with requests.get(url, stream=True, verify=False) as r:
        with open(path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)


def load_cifar10(normalize=True, download_dir=""):
    """
    Download Cifar10 training data from `keras.datasets.load_cifar10`

    :param normalize: whether or not to normalize data
    :type normalize: bool
    :return: 2 tuples containing training and testing data respectively
    :rtype (`np.ndarray`, `np.ndarray`), (`np.ndarray`, `np.ndarray`)
    :param download_dir: directory to download data
    :type download_dir: `str`
    """
    from keras.datasets import cifar10
    local_file = os.path.join(download_dir, "cifar10.npz")
    # print('local file: ', local_file)
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')
    train_y = train_y.flatten()
    test_y = test_y.flatten()
    if normalize:
        train_x = train_x / 255
        test_x = test_x / 255

    # save cifar10 dataset
    np.savez(local_file, x_train=train_x, y_train=train_y,
             x_test=test_x, y_test=test_y)
    return (train_x, train_y), (test_x, test_y)


def load_mnist(normalize=True, download_dir=""):
    """
    Download MNIST training data from source used in `keras.datasets.load_mnist`
    :param normalize: whether or not to normalize data
    :type normalize: bool
    :param download_dir: directory to download data
    :type download_dir: `str`
    :return: 2 tuples containing training and testing data respectively
    :rtype (`np.ndarray`, `np.ndarray`), (`np.ndarray`, `np.ndarray`)
    """
    local_file = os.path.join(download_dir, "mnist.npz")
    if not os.path.isfile(local_file):
        save_file(local_file, "https://s3.amazonaws.com/img-datasets/mnist.npz")

        with np.load(local_file, allow_pickle=True) as mnist:
            x_train, y_train = mnist['x_train'], mnist['y_train']
            x_test, y_test = mnist['x_test'], mnist['y_test']
            if normalize:
                x_train = x_train.astype('float32')
                x_test = x_test.astype('float32')

                x_train /= 255
                x_test /= 255

        # save the normalized mnist.npz
        np.savez(local_file, x_train=x_train, y_train=y_train,
                 x_test=x_test, y_test=y_test)
    else:
        with np.load(local_file, allow_pickle=True) as mnist:
            x_train, y_train = mnist['x_train'], mnist['y_train']
            x_test, y_test = mnist['x_test'], mnist['y_test']

    return (x_train, y_train), (x_test, y_test)


def load_nursery(download_dir=""):
    """
    Download Nursery training data set

    :param download_dir: directory to download data
    :type download_dir: `str`
    :return: training dataset for data set
    :rtype: `pandas.core.frame.DataFrame`
    """
    local_file = os.path.join(download_dir, "nursery.data")
    if not os.path.isfile(local_file):
        save_file(
            local_file, "https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data")
    training_dataset = pd.read_csv(local_file, dtype='category', header=None)
    # url data does not have label for each column,
    # so insert column labels here.
    training_dataset.columns = ['1', '2', '3', '4', '5', '6', '7',
                                '8', 'class']
    return training_dataset


def load_adult(download_dir=""):
    """
    Download Adult training data set,
    and perform the following pre-processing steps:
        * Drop `fnlwgt` feature
        * Add column labels ['1', '2', '3'...'13', 'class']

    :param download_dir: directory to download data
    :type download_dir: `str`
    :return: training dataset for data set
    :rtype: `pandas.core.frame.DataFrame`
    """
    local_file = os.path.join(download_dir, "adult.data")
    if not os.path.isfile(local_file):
        save_file(
            local_file, "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data")
    training_dataset = pd.read_csv(local_file, dtype='category', header=None)

    # drop 'fnlwgt' column
    training_dataset = training_dataset.drop(
        training_dataset.columns[2], axis='columns')

    # url data does not have label for each column,
    # so insert column labels here.
    training_dataset.columns = ['1', '2', '3', '4', '5', '6', '7',
                                '8', '9', '10', '11', '12', '13', 'class']

    return training_dataset

def load_german(download_dir=""):
    """
    Download German Credit Scoring training data set, and add column labels ['1', '2', '3'...'13', 'class']
    :param download_dir: directory to download data
    :type download_dir: `str`
    :return: training dataset for data set
    :rtype: `pandas.core.frame.DataFrame`
    """
    local_file = os.path.join(download_dir, "german.data")
    if not os.path.isfile(local_file):
        save_file(
            local_file, "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")
    training_dataset = pd.read_csv(
        local_file, sep=' ', dtype='category', header=None)
    # url data does not have label for each column,
    # so insert column labels here.
    training_dataset.columns = ['1', '2', '3', '4', '5', '6', '7',
                                '8', '9', '10', '11', '12', '13', '14',
                                '15', '16', '17', '18', '19', '20', 'class']
    return training_dataset


def load_compas(download_dir=""):
    """
    Download Compas (ProPublica recidivism) training data set and rename 'two_year_recid' to 'class':
    :param download_dir: directory to download data
    :type download_dir: `str`
    :return: training dataset for data set
    :rtype: `pandas.core.frame.DataFrame`
    """
    local_file = os.path.join(download_dir, "compas-scores-two-years.csv")
    if not os.path.isfile(local_file):
        save_file(
            local_file, "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv")
    with open(local_file, 'rb') as fd:
        compas_unzip = gzip.GzipFile(fileobj=fd)
        training_dataset = pd.read_csv(compas_unzip)

    training_dataset['class'] = training_dataset['two_year_recid']
    training_dataset = training_dataset.drop('two_year_recid', axis=1)

    return training_dataset


def load_binovf():
    """
    Generate Binary Classification Overfit - Based on features where all 0 class
    label values are derived with 0 features and similarly for 1 class labels.
    Generates a sample of 1,000 data points with univariate features, 500 for
    each class.

    :return: Generated training dataset for data set for Binary Overfit set.
    :rtype: `tuple` of (`np.ndarray`, `np.ndarray`)
    """
    X, y = np.zeros((1000, 1)), np.zeros((1000))
    X[500:, :] = np.ones((500, 1))
    y[500:] = np.ones((500))

    return X, y


def load_multovf():
    """
    Generate Multiclass Classification Overfit - Based on features where all 0 class
    label values are derived with 0 features and similarly for 1 class labels.
    Generates a sample of 1,000 data points with univariate features, 500 for
    each class.

    :return: Generated training dataset for data set for Multiclass Overfit set.
    :rtype: `tuple` of (`np.ndarray`, `np.ndarray`)
    """
    X, y = np.zeros((1000, 3)), np.zeros((1000))
    for i, x in enumerate(range(200, 1000, 200)):
        X[x:x+200, :] = np.ones((200, 3)) * i
        y[x:x+200] = np.ones((200)) * i

    return X, y


def load_linovf():
    """
    Generate RegressionOverfit Data - Based on a 1 to 1 linear relationship of the
    input to the output values. Generates a sample of 1,000 data points with
    univariate features. (Models a perfect y = x relationship)

    :return: Generated training dataset for data set for Linear Overfit set.
    :rtype: `tuple` of (`np.ndarray`, `np.ndarray`)
    """
    data = np.random.uniform(0, 100, 1000)
    X, y = data, data
    return X, y


def load_higgs(download_dir=""):
    """
    Download Higgs Boson training dataset, and append column labels.

    :param download_dir: directory to download data
    :type download_dir: `str`
    :return: Training dataset for data set.
    :rtype: `tuple` of (`np.ndarray`, `np.ndarray`)
    """
    # Define Local Donwload Directory
    local_file = os.path.join(download_dir, "HIGGS.csv.gz")

    # Download File If Not Present in System
    if not os.path.isfile(local_file):
        print('Dataset not available in directory, downloading from source...')
        save_file(
            local_file, 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz')

    # Load Dataset
    print('Loading dataset from ' + local_file)
    data = pd.read_csv(local_file, compression='gzip', header=None).to_numpy()

    # Parse Column Features and Target and Return Values
    return data[:, 1:], data[:, 0]


def load_airline(download_dir=""):
    """
    Download Airline Arrivals Dataset. To simplify data preprocessing, we
    perform this here instead at the DataLoader end to offset the process.
    Preprocessing includes: i) dropping data leaked features, ii) categorical
    encoding, iii) dropping features, and iv) one hot encoding.

    :param download_dir: directory to download data
    :type download_dir: `str`
    :return: Training dataset for data set.
    :rtype: `tuple` of (`np.ndarray`, `np.ndarray`)
    """
    # Define Local Donwload Directory
    local_file = os.path.join(download_dir, "DelayedFlights.csv")

    # Download File If Not Present in System
    if not os.path.isfile(local_file):
        print('Please download the csv file directly from the source ' +
              'https://www.kaggle.com/giovamata/airlinedelaycauses and extract ' +
              'and save the csv file under the examples/datasets/ directory.')
        sys.exit()

    # Load Dataset
    print('Loading dataset from ' + local_file)
    data = pd.read_csv(local_file)

    # Define Auxillary Function
    def get_dtypes(data, features):
        output = {}
        for f in features:
            dtype = str(data[f].dtype)
            if dtype not in output.keys():
                output[dtype] = [f]
            else:
                output[dtype] += [f]
        return output

    # Drop Unecessary & Leaky Features
    data = data.drop("Unnamed: 0", 1)

    target = ["Cancelled"]
    leaky_features = ["Year", "Diverted", "ArrTime", "ActualElapsedTime",
                      "AirTime", "ActualElapsedTime", "AirTime", "ArrDelay",
                      "TaxiIn", "CarrierDelay", "WeatherDelay", "NASDelay",
                      "SecurityDelay", "LateAircraftDelay", "CancellationCode"]

    features = [x for x in data.columns if (x != target[0]) & (x not in
                                                               leaky_features) & (len(data[x].unique().tolist()) > 1)]

    data = data[data["Month"].isin([10, 11, 12])]

    dtypes = get_dtypes(data, features)

    # Categorical and Numerical Feature Selection
    categories = ["Month", "DayOfWeek", "DayofMonth"]
    categories += dtypes["object"]
    numerics = [i for i in dtypes["int64"] if i not in categories]
    numerics += dtypes["float64"]

    for numeric in numerics:
        data[numeric] = data[numeric].fillna(0)
    categories.remove("TailNum")

    cancelled = data[data[target[0]] == 1]
    not_cancelled = data[data[target[0]] == 0]

    data = pd.concat([cancelled, not_cancelled.sample(n=len(cancelled))], 0)

    one_hot_encoded = pd.get_dummies(data[categories].fillna("Unknown"))
    X = pd.concat([one_hot_encoded, data[numerics].fillna(0)], 1)
    y = data[target[0]]

    return X.to_numpy(), y.to_numpy()


def load_diabetes(download_dir=""):
    """
    Download Diabetese Dataset from source.  To simplify data preprocessing, we
    perform this here instead at the DataLoader end to offset the process.
    Preprocessing includes primarily feature dropping.

    :param download_dir: directory to download data
    :type download_dir: `str`
    :return: training dataset for data set
    :rtype: `np.ndarray`, `np.ndarray`
    """
    # Define Local Donwload Directory
    local_file = os.path.join(download_dir, "diabetic_data.csv")

    # Download File If Not Present in System
    if not os.path.isfile(local_file):
        # Download Raw Data
        print('Dataset not available in directory, downloading from source...')
        zip_dir = os.path.join(download_dir, 'dataset_diabetes.zip')
        save_file(
            zip_dir, 'https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip')

        # Extract Content
        zip = ZipFile(os.path.join(download_dir, 'dataset_diabetes.zip'))
        zip.extractall(download_dir)
        zip.close()

        # Cleanup File Directory
        os.rename(os.path.join(download_dir, 'dataset_diabetes/diabetic_data.csv'),
                  os.path.join(download_dir, 'diabetic_data.csv'))
        os.remove(zip_dir)
        shutil.rmtree(os.path.join(download_dir, 'dataset_diabetes/'))

    # Load Dataset
    print('Loading dataset from ' + local_file)
    data = pd.read_csv(local_file)

    # Drop Features
    drop_id = ['encounter_id', 'patient_nbr', 'race', 'weight', 'payer_code',
               'medical_specialty', 'diag_1', 'diag_2', 'diag_3']
    data = data.drop(drop_id, axis=1)

    return data


def load_leaf_femnist(download_dir="", orig_dist=False):
    """
    Load FEMNIST Data from LEAF.  To simplify data preprocessing, we
    perform this here instead at the DataLoader end to offset the process.
    Preprocessing includes primarily feature dropping.

    :param download_dir: directory to download data
    :type download_dir: `str`
    :return: training, testing dataset
    :rtype: `np.ndarray`, `np.ndarray`
    """
    # additional imports for LEAF FEMNIST
    import hashlib
    from PIL import Image

    def read_femnist_data(dir, orig_dist):
        """
        Load data from files in given directory.

        :param dir: directory containing the raw uncompressed FEMNIST data
        :type dir: `str`
        :param orig_dist: decides whether to split the data up using FEMNIST's distribution or custom.
        :type orig_dist: `bool`
        :return: training, testing dataset
        :rtype: `np.ndarray`, `np.ndarray`
        """

        files = [f for f in os.listdir(dir) if f.endswith(".json")]
        x_train, y_train, x_test, y_test = [], [], [], []
        partywise_data = {}
        for i, f in enumerate(files):
            print("Loading", f)
            with open(os.path.join(dir, f), 'r') as fp:
                all_data = json.load(fp)
            for party_id, data in all_data['user_data'].items():
                if orig_dist:
                    partywise_data[party_id] = data
                else:
                    num_test_samples = int(len(data['x']) * 0.1)
                    x_train.extend(data['x'][:-num_test_samples])
                    y_train.extend(data['y'][:-num_test_samples])
                    x_test.extend(data['x'][-num_test_samples:])
                    y_test.extend(data['y'][-num_test_samples:])

        if orig_dist :
            print("Using FEMNIST's Original Dist")
            print("# of Parties : ", len(list(partywise_data.keys())))
            return partywise_data
        else :
            print("Dataset Size -> train:", len(x_train), 'test', len(x_test))
            return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))

    if os.path.exists(os.path.join(download_dir, 'all_data')):
        print("FEMNIST Data has been preprocessed, loading it....")
        return read_femnist_data(os.path.join(download_dir, 'all_data'), orig_dist)

    # Download Dataset URLs
    FEMNIST_BY_CLASS =  "https://s3.amazonaws.com/nist-srd/SD19/by_class.zip"
    FEMNIST_BY_WRITER = "https://s3.amazonaws.com/nist-srd/SD19/by_write.zip"
    by_class_zip_file = os.path.join(download_dir, "by_class.zip")
    by_writer_zip_file = os.path.join(download_dir, "by_write.zip")

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # Download Data/Metada files and extract intor download_dir
    if not os.path.isfile(by_writer_zip_file):
        print("Downloading by_write.zip")
        save_file(by_writer_zip_file, FEMNIST_BY_WRITER)
        print("Extracting by_write.zip")
        zip = ZipFile(by_writer_zip_file)
        zip.extractall(os.path.join(download_dir))
        zip.close()
        os.remove(by_writer_zip_file)
        print("Completed extracting by_write.zip")

    if not os.path.isfile(by_class_zip_file):
        print("Downloading by_class.zip")
        save_file(by_class_zip_file, FEMNIST_BY_CLASS)
        print("Extracting by_class.zip")
        zip = ZipFile(by_class_zip_file)
        zip.extractall(os.path.join(download_dir))
        zip.close()
        os.remove(by_class_zip_file)
        print("Completed extracting by_class.zip")

    # PREPROCESSING START - Raw data to readable and distributed .json files
    # Read Class Directories
    print("Mapping Directories.....")
    class_files = []  # (class, file directory)
    class_dir = os.path.join(download_dir, 'by_class')
    rel_class_dir = os.path.join('by_class')
    classes = os.listdir(class_dir)
    classes = [c for c in classes if len(c) == 2]

    for cl in classes:
        cldir = os.path.join(class_dir, cl)
        rel_cldir = os.path.join(rel_class_dir, cl)
        subcls = os.listdir(cldir)
        subcls = [s for s in subcls if (('hsf' in s) and ('mit' not in s))]
        for subcl in subcls:
            subcldir = os.path.join(cldir, subcl)
            rel_subcldir = os.path.join(rel_cldir, subcl)
            images = os.listdir(subcldir)
            image_dirs = [os.path.join(rel_subcldir, i) for i in images]
            for image_dir in image_dirs:
                class_files.append((cl, image_dir))

    # Read Writer Directories
    write_files = []  # (writer, file directory)
    write_dir = os.path.join(download_dir, 'by_write')
    rel_write_dir = os.path.join('by_write')
    write_parts = os.listdir(write_dir)

    for write_part in write_parts:
        writers_dir = os.path.join(write_dir, write_part)
        rel_writers_dir = os.path.join(rel_write_dir, write_part)
        writers = os.listdir(writers_dir)
        for writer in writers:
            writer_dir = os.path.join(writers_dir, writer)
            rel_writer_dir = os.path.join(rel_writers_dir, writer)
            wtypes = os.listdir(writer_dir)
            for wtype in wtypes:
                type_dir = os.path.join(writer_dir, wtype)
                rel_type_dir = os.path.join(rel_writer_dir, wtype)
                images = os.listdir(type_dir)
                image_dirs = [os.path.join(rel_type_dir, i) for i in images]
                for image_dir in image_dirs:
                    write_files.append((writer, image_dir))

    print("Done mapping directories")

    # Get Hash -> Target Label Conversion
    class_file_hashes = []
    count = 0
    for tup in class_files:
        if (count % 100000 == 0):
            print('hashed %d class images' % count)
        (cclass, cfile) = tup
        file_path = os.path.join(download_dir, cfile)
        chash = hashlib.md5(open(file_path, 'rb').read()).hexdigest()
        class_file_hashes.append((cclass, cfile, chash))
        count += 1
    write_file_hashes = []
    count = 0
    for tup in write_files:
        if (count % 100000 == 0):
            print('hashed %d write images' % count)
        (cclass, cfile) = tup
        file_path = os.path.join(download_dir, cfile)
        chash = hashlib.md5(open(file_path, 'rb').read()).hexdigest()
        write_file_hashes.append((cclass, cfile, chash))
        count += 1

    class_hash_dict = {}
    for i in range(len(class_file_hashes)):
        (c, f, h) = class_file_hashes[len(class_file_hashes)-i-1]
        class_hash_dict[h] = (c, f)
    write_classes = []
    for tup in write_file_hashes:
        (w, f, h) = tup
        write_classes.append((w, f, class_hash_dict[h][0]))

    # Generate List of Writers
    writers = [] # each entry is a (writer, [list of (file, class)]) tuple
    cimages = []
    (cw, _, _) = write_classes[0]
    for (w, f, c) in write_classes:
        if w != cw:
            writers.append((cw, cimages))
            cw = w
            cimages = [(f, c)]
        cimages.append((f, c))
    writers.append((cw, cimages))

    users = []
    num_samples = []
    user_data = {}

    all_data_path = os.path.join(download_dir, 'all_data')
    if not os.path.exists(all_data_path):
        os.makedirs(all_data_path)

    writer_count = 0
    json_index = 0

    # Preprocess images and save to .json
    for (w, l) in writers:

        users.append(w)
        num_samples.append(len(l))
        user_data[w] = {'x': [], 'y': []}

        size = 28, 28  # original image size is 128, 128
        for (f, c) in l:
            file_path = os.path.join(download_dir, f)
            img = Image.open(file_path)
            gray = img.convert('L')
            gray.thumbnail(size, Image.ANTIALIAS)
            arr = np.asarray(gray).copy()
            vec = arr.flatten()
            vec = vec / 255  # scale all pixel values to between 0 and 1
            vec = vec.tolist()

            # Relabeling
            if c.isdigit() and int(c) < 40:
                nc = (int(c) - 30)
            elif int(c, 16) <= 90: # uppercase
                nc = (int(c, 16) - 55)
            else:
                nc = (int(c, 16) - 61)
            user_data[w]['x'].append(vec)
            user_data[w]['y'].append(nc)

        writer_count += 1
        if writer_count == 100: # No more than 100 writers in a file

            all_data = {}
            all_data['users'] = users
            all_data['num_samples'] = num_samples
            all_data['user_data'] = user_data

            file_name = 'all_data_%d.json' % json_index
            file_path = os.path.join(all_data_path, file_name)

            print('writing %s' % file_name)

            with open(file_path, 'w') as outfile:
                json.dump(all_data, outfile)

            writer_count = 0
            json_index += 1

            users[:] = []
            num_samples[:] = []
            user_data.clear()
    # PREPROCESSING END

    return read_femnist_data(all_data_path, orig_dist)


def load_simulated_federated_clustering(**kwargs):
    r"""
    Generates a simulated federated clustering dataset as described in
    [https://arxiv.org/abs/1911.00218]. `L` true global centroids generate a
    `D` dimensional data partitioned in `J` clients.
    :param \**kwargs
            See below
    :Keyword Arguments:
        * *L* (``int``) -- Number of true global centroids, default 50
        * *J* (``int``) -- Number of clients, default 10
        * *D* (``int``) -- Data dimension, default 50
        * *M* (``int``) -- Data points per group, default 1000
        * *mu0* (``float``) -- Global mean, default 0.0
        * *global_sd* (``float``) -- Global standard deviation, default `np.sqrt(L)`
        * *local_sd* (``float``) -- Local standard deviation, default 1.0
    :return: generated data of shape (J, M, D)
    :rtype: `np.ndarray`
    """

    # Number of global centroids
    L = kwargs.get('L', 50)
    J = kwargs.get('J', 10)                             # Number of clients
    D = kwargs.get('D', 50)                             # Data dimension
    # data points per group
    M = kwargs.get('M', 1000)
    mu0 = kwargs.get('mu0', 0)                          # Global mean
    # Global standard deviation
    global_sd = kwargs.get('global_sd', np.sqrt(L))
    # Local standard deviation
    local_sd = kwargs.get('local_sd', 1.0)

    a, b = 1, 1
    global_p = np.random.beta(a=a, b=b, size=L)
    global_atoms = np.random.normal(
        mu0, global_sd, size=(L, D))    # Global set of centroids

    data = []       # data will be (J, M, D) dimensional array
    used_components = set()
    atoms = []

    for j in range(J):
        atoms_j_idx = [l for l in range(
            L) if np.random.binomial(1, global_p[l])]
        used_components.update(atoms_j_idx)
        atoms_j = np.random.normal(global_atoms[atoms_j_idx], scale=local_sd)

        # Generating gaussian mixture
        K = atoms_j.shape[0]
        mixing_prop = np.random.dirichlet(np.ones(K))
        assignments = np.random.choice(K, size=M, p=mixing_prop)
        data_j = np.zeros((M, D))

        for k in range(K):
            data_j[assignments == k] = np.random.normal(
                loc=atoms_j[k], scale=0.1, size=((assignments == k).sum(), D))

        data.append(data_j)
        atoms.append(atoms_j)   # Local set of centroids

    return data


def load_wikipedia(num_examples=500, starting_index=0):
    """
    Download 2017 Wikipedia dataset from gensim. To simplify data preprocessing, we
    perform this here instead of at the DataLoader end to offset the process

    :return: Training dataset for data handler
    :rtype: 'list
    """
    try:
        corpus = api.load("wiki-english-20171001")
    except Exception as e:
        print(e)
        print("must download dataset first. attempting to download...")
        print(api.load("wiki-english-20171001", return_path=True))
        corpus = api.load("wiki-english-20171001")

    tagged_documents = []
    i = 0
    sample_count = 0
    for doc in corpus:
        if i < starting_index:
            i += 1
            continue

        full_text = ""

        for texts in doc['section_texts']:
            full_text += texts + '\n'

        tokens = simple_preprocess(full_text)
        tagged_documents.append(TaggedDocument(words=tokens, tags=[doc['title']]))

        i += 1
        sample_count += 1
        if sample_count == num_examples:
            break

    return tagged_documents

def load_ionosphere(download_dir=""):
    """
    Download Ionosphere training data set

    :param download_dir: directory to download data
    :type download_dir: `str`
    :return: training dataset for data set
    :rtype: `pandas.core.frame.DataFrame`
    """
    local_file = os.path.join(download_dir, "ionosphere.data")
    if not os.path.isfile(local_file):
        save_file(
            local_file, "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data")
    dataset = pd.read_csv(local_file, dtype='category', header=None)

    header = ['feature-{}'.format(i+1) for i in range(34)]
    header.append('class')
    dataset.columns = header

    return dataset

def load_openml_dataset(download_dir="", dataset_name="ailerons", active=False):
    """
    Download Ionosphere training data set

    :param download_dir: directory to download data
    :type download_dir: `str`
    :param dataset_name: name of dataset .csv file
    :type dataset_name: `str`
    :return: training dataset for data set
    :rtype: `pandas.core.frame.DataFrame`
    """
    local_file = os.path.join(download_dir, f"{dataset_name}.csv")
    if not os.path.isfile(local_file):
        print(f"Dataset {local_file} not found")
        return None
    dataset = pd.read_csv(local_file, dtype='category', header=0)

    header = ['feature-{}'.format(i+1) for i in range(len(dataset.columns)-1)]
    header.append('class')
    dataset.columns = header

    labels = dataset['class'].unique()
    label_nums = np.arange(0,len(labels),1,dtype=int)

    replace_dict = {labels[i]: label_nums[i] for i in range(len(labels))}
    dataset['class'] = dataset['class'].replace(replace_dict)

    return dataset
