- [x] implement file upload mechanism
- [x] implement file type checking mechanism
- [x] implement file validity checking for pdf (ex. pdf with no meaningful text)
- [] implement file validty checking for csv
- [] display what the application expects in terms of files
- [x] implement pdf text extraction methods
- [x] implement pdf text extraction unit tests
- [x] implement pdf exploratory data analysis methods
- [x] implement pdf exploratory data analysis unit tests
- [x] implement zero shot classification
- [x] implement zero shot classification unit testing
- [x] implement summarization transformer
- [x] implement summarization unit testing
- [] implement keyphrase extraction methods
- [] implement keyphrase unit testing
- [x] implement NER methods
- [] implement NER testing
- [] implement sentiment analysis methods
- [] implement sentiment analysis unit testing
- [] implement q&a query
- [] implement q&a query unit testing
- [] implement a checkbox function that allows users to select which auto NLP methods they want done
- [] csv functions...
- [] optimize zero shot classification accuracy & speed (distributed computing, fine tuned models, sentence chunk size, chunk blending, etc.)
- [] optimize summarization accuracy and speed
- [] add top bi_grams and tri_grams (no stop words) as an eda result
- [] organize ml files into folders
- [] specify that the applications only supposed to be used with English text
- [] figure out the max token length for each model, and use that to split up things.
- 







   Type of File
   CSV - Clustering, Regression, Classification
   PDF - Summarization, Keyword Extraction, Zero-Shot Classifier, Q&A Bot
   Glossary
   Push to Github
   CI/CD Pipeline, testing triggered on each commit to github (test all the tasks using samples, as well as page navigation)
   Deployment to streamlit site
   Web App Styling/Semantics (main page persist, resetting, )
   Specify that this is currently for english only, maybe add multimodal later
   Make sure in the embeddings portion, go column by column, and if its non numeric then embed it. Else dont embed it. This is in case we get text + numeric features
   expecting a cleaned .csv, with the exception of null values (so labeled columns, and same-type columns)
   add checkboxes for what the user wants to see in the auto report (ex. for kmeans, have checkboxes for silhoulette, wcss, elbow plot, scatter plot, etc.)
   do the sidebar stuff last, make the output a button on the sidebar
   sections (upload, eda, nlp, downloads, glossary)
   maybe implement start over button
   add something that prevents user from uploading an empty .pdf file
   prompt it only handles english words.
   optimize runtime for the zero-shot classification (model size, sentence chunk size)
   optimize accuracy of zero shot (blending chunks together to hold the context)
