import os

def install_nlp_resources():
    os.system("python -m textblob.download_corpora --verbose")

if __name__ == "__main__":
    install_nlp_resources()
