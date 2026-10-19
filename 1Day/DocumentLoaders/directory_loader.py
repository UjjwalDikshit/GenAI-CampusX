# here we need to tell about folder path, glob, tools to load file

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path = 'document',
    glob = '*.pdf',
    loader_cls = PyPDFLoader
)

# docs = loader.load()


# print(len(docs)) #11

# print(docs[0].page_content, docs[0].metadata)
# print(docs[7].page_content, docs[7].metadata)
# print(docs[10].page_content, docs[10].metadata)

#------------------------------------------------------------

docs0 = loader.load()
docs1 = loader.lazy_load()


# for document in docs0:  # takes too much time, as have to load everyhitng into ram then work
#     print(document.metadata)

for document in docs1:  # memory me lao, print kro hatao, then 2nd memory me lao,....continue till last
    print(document.metadata)

