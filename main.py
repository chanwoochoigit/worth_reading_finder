import sys
from utils import take_input
from analyse_document_bert import analyse

def main(json_string):
	return analyse(json_string)

if __name__ == '__main__':
	print(sys.argv)
	input_json = sys.argv[1:2]
	print(input_json)
	print(main(input_json[0]))
