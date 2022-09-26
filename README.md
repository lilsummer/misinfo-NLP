# misinfo-NLP
ensemble bert to identify misinformation in public health

Example use:

curl \
--header "Content-Type: application/json" \
--request POST \
--data '{"tweets": ["!", "Covid is fake", "My name is Linh", "President Trump's comments about the coronavirus death rate were 100% correct. The media falsely claimed he was spreading misinformation. They falsely reported that his comments weren't in line with top health officials. That was 100% fake news."]}' \
http://3.90.50.113:8000/predict