# Collaborative-Filtering-Clasification-based-approach
A simple classification based approach for addressing user and  item cold-start problem

## Approach

1. Construct User vector and Item vector from available metadata. 
2. Concatenate the two vectors to get one single user-item vector.
3. Instead of using neighbourhood based approach, we use classification. 
4. Given a user-tem vector, we then try to predict(Classify) which class does it belong to, where class refers to the rating.
