# Notmnist
if [[ ! -d notMNIST_small ]]; then
    echo '### NOTMNIST'
    wget http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz
    tar -xzf notMNIST_small.tar.gz
    rm notMNIST_small.tar.gz
    # Remove two corrupted files
    rm notMNIST_small/A/RGVtb2NyYXRpY2FCb2xkT2xkc3R5bGUgQm9sZC50dGY=.png
    rm notMNIST_small/F/Q3Jvc3NvdmVyIEJvbGRPYmxpcXVlLnR0Zg==.png
fi

# Omniglot
if [[ ! -d omniglot ]]; then
    echo '### OMNIGLOT'
    mkdir omniglot
    wget https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip
    wget https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip
    unzip images_evaluation.zip
    unzip images_background.zip
    mv images_evaluation/* omniglot/
    mv images_background/* omniglot/
    rm -r images_background
    rm -r images_evaluation
    rm images_evaluation.zip
    rm images_background.zip
fi

# LFW-a
if [[ ! -d lfw-a ]]; then
    echo '### LFW-A'
    wget http://vis-www.cs.umass.edu/lfw/lfw-a.tgz
    tar -xzf lfw-a.tgz
    mv lfw lfw-a
    rm lfw-a.tgz
fi

mkdir -p nonmovies
cd nonmovies

# Reuters 21-578
if [[ ! -d reuters21578 ]]; then
    echo '### REUTERS'
    wget http://www.daviddlewis.com/resources/testcollections/reuters21578/reuters21578.tar.gz
    mkdir reuters21578
    tar -xzf reuters21578.tar.gz -C reuters21578
    rm reuters21578.tar.gz
fi

# Customer review dataset
if [[ ! -d 'customer review data' ]]; then
    echo '### Customer review dataset'
    wget http://www.cs.uic.edu/~liub/FBS/CustomerReviewData.zip
    unzip CustomerReviewData.zip
    rm CustomerReviewData.zip
fi

# Polarity
if [[ ! -d review_polarity ]]; then
    echo '### Polarity'
    wget http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz
    mkdir review_polarity
    tar -xzf review_polarity.tar.gz -C review_polarity
    rm review_polarity.tar.gz
fi

# 20 news
if [[ ! -d 20news-18828 ]]; then
    echo '### 20 news'
    wget http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz
    tar -xzf 20news-18828.tar.gz
    rm 20news-18828.tar.gz
fi

cd ../
mkdir -p movies
cd movies

# Movies
if [[ ! -d aclImdb ]]; then
    wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    tar -xzvf aclImdb_v1.tar.gz
    rm aclImdb_v1.tar.gz
fi

# Embedding
if [[ ! -d glove.6B ]]; then
    wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
    unzip glove.6B.zip
    rm glove.6B.zip
fi