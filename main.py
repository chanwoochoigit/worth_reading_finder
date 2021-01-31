import sys
from utils import take_input
from analyse_document_bert import analyse

if __name__ == '__main__':
    with open("frame.json") as file:
        json_string = file.read()

    print(analyse(json_string))