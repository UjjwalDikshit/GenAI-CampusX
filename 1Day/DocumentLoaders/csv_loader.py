from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='Social_Network_Ads.csv')

data = loader.load() # can also be load by lazy loader.

print(data[0])