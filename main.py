import sys
from utils import take_input
from analyse_document_bert import analyse

def main(json_string):
    json_object = take_input(json_string)
    return analyse(json_object=json_object)

if __name__ == '__main__':
    with open("frame.json") as file:
        json_string = file.read()
    print(main(json_string))