
library(keras)
library(textstem)
library(stopwords)
library(text2vec)
#load data set
data("movie_review")
#Read training set

summary(movie_review)


#Assign to new variable
review_data = movie_review$review

#Function to remove stop words and digits and lemmatize
cleanReview <- function (review){
  for(i in nrow(review)){
    #remove numbers
    text_review = gsub('[[:digit:]]+', '', review[i])
    #remove stop words
    text_review = stopwords(text_review)
    #lemmatize
    word_vector = str_split(text_review," ")
    word_vector = lemmatize_words(word_vector)
    review[i] = paste(word_vector, collapse=" ")
  }
  return(review)
}

#First lets do some pre processing
#text_tokenizer already removes punctuation and converts to lower case
review_data = cleanReview(review_data)


max_words <-10000
review_tokenizer <-text_tokenizer(num_words =max_words)%>%fit_text_tokenizer(review_data)
review_sequences <-texts_to_sequences(review_tokenizer, review_data)
# maximum length of a sequence. This will be the same for all
maxlen <-1000
review_padded_train <-pad_sequences(review_sequences,maxlen =maxlen)

#rating labels
labels <-as.array(movie_review$sentiment)


embedding_dim <-32

#Model using LSTM
#Please see https://keras.rstudio.com/reference/layer_lstm.html
model <-keras_model_sequential()%>%
  layer_embedding(input_dim =max_words,output_dim =embedding_dim)%>%
  layer_lstm(units = 32) %>%
  layer_dense(units =1,activation ="sigmoid")%>%
  compile(optimizer =optimizer_rmsprop(),loss ="binary_crossentropy",metrics ="accuracy")
#Go grab a coffee, this will take ages
fit <-model%>%fit(review_padded_train, 
                  labels,
                  batch_size =nrow(review_padded_train)*0.01,
                  epochs =20,
                  callbacks = list(
                    callback_early_stopping(monitor = "val_loss", patience = 5,restore_best_weights = TRUE)),
                  validation_split = 0.2)

#Check metrics
fit$metrics