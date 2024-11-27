from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import numpy as np
import joblib
import os
from .serializers import InsuranceSerializer

#get the path to the pickled model file
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','Model','best_model.pkl')

# load the pickled model
model = joblib.load(model_path)


@api_view(['POST'])
def predict(request):
    if request.method == 'POST':
        # deserialize the input data from the request
        serializer = InsuranceSerializer(data=request.data)
        if serializer.is_valid():
            #  convert input data to input format for model
            input_data = tuple(serializer.validated_data.values())
            input_data_as_numpy_array = np.asarray(input_data)
            input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
            print(input_data_reshaped)
        #make a prediction using the model
        prediction = model.predict(input_data_reshaped)

        #return the prediction as a JSON response
        return Response(prediction)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rest_framework.response import Response
from rest_framework.decorators import api_view
import numpy as np

# Define predefined questions and responses
predefined_questions = [
    "What is medical insurance?",
    "How can I calculate medical insurance costs?",
    "What factors affect medical insurance premiums?",
    "How can I use this prediction system?",
    "Can I download the insurance prediction report?",
    "What is BMI?",
    "Is smoking considered in the insurance cost?",
    "How accurate is the cost prediction?",
    "What does this chatbot do?",
    "How do I reset my password?",
    "Do you store my personal data?",
    "Can I update my profile information?",
    "What is the minimum age for insurance coverage?",
    "How do I contact support?",
    "How does the system use my data?",
    "Is the prediction system free to use?",
    "What if I enter incorrect details?",
    "Is the chatbot available 24/7?",
    "Can I get a discount on my premium?",
    "What happens if I stop smoking?",
    "How does smoking affect my medical insurance cost?",
    "Why do smokers pay more for medical insurance?",
    "What diseases caused by smoking raise insurance costs?",
    "Can quitting smoking lower my insurance premiums?"
]


predefined_responses = [
    "Medical insurance covers the cost of an individual's medical and surgical expenses.",
    "You can use our system by entering details like age, BMI, and other factors to predict costs.",
    "Factors like age, gender, BMI, smoking habits, and number of dependents affect premiums.",
    "Simply log in, provide the required details, and get your insurance cost prediction.",
    "Yes, you can download the report as a PDF after generating the prediction.",
    "BMI stands for Body Mass Index, a measure of body fat based on height and weight.",
    "Yes, smoking habits significantly impact medical insurance costs.",
    "The predictions are based on a trained machine learning model and are quite accurate.",
    "This chatbot answers your queries related to medical insurance and the prediction system.",
    "Go to the login page and click on 'Forgot Password' to reset it.",
    "Your data is securely stored and used only for prediction purposes.",
    "Yes, you can update your profile information in the 'My Account' section.",
    "The minimum age is usually 18 years, but it may vary by provider.",
    "You can reach our support team via email at support@insurancepredictor.com.",
    "The system uses your data to predict insurance costs based on statistical models.",
    "Yes, the prediction system is free to use for registered users.",
    "You can edit your input and re-calculate the insurance cost.",
    "Yes, the chatbot is available to answer your queries at any time.",
    "Discounts depend on the insurance provider's policy and your health profile.",
    "If you stop smoking, your insurance premiums might decrease over time.",
    "Smoking increases the risk of health issues, which leads to higher insurance premiums.",
    "Smokers are more likely to develop health conditions like lung disease, increasing their risk and cost to insurers.",
    "Diseases such as lung cancer, heart disease, and chronic respiratory issues raise costs.",
    "Yes, many insurers offer lower premiums if you maintain a smoke-free status for a specified period."
]


# Create the TF-IDF Vectorizer and fit it on the predefined questions
# Passing stop_words='english' will ignore common stopwords
vectorizer = TfidfVectorizer(stop_words='english').fit(predefined_questions)

@api_view(['POST'])
def chatbot_response(request):
    user_input = request.data.get('query')

    if not user_input:
        return Response({"error": "No query provided"}, status=400)

    # Add the user's query to the predefined questions
    queries = predefined_questions + [user_input]

    # Transform both predefined questions and user input to vectors
    vectors = vectorizer.transform(queries)

    # Compute the cosine similarity between user input and predefined questions
    similarity_matrix = cosine_similarity(vectors[-1], vectors[:-1])
    
    # Get the maximum similarity value and its index
    max_similarity = similarity_matrix.max()
    most_similar_index = np.argmax(similarity_matrix)

    # Define a threshold for similarity
    similarity_threshold = 0.7  # You can change this value

    if max_similarity < similarity_threshold:
        # If the similarity is below the threshold, return a fallback response
        return Response({"response": "I'm sorry, I didn't understand that. Can you rephrase your question?"})

    # Return the corresponding response
    return Response({"response": predefined_responses[most_similar_index]})

