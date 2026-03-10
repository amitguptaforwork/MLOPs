import streamlit as st

st.title("Spam Detection App")  
st.write("Enter a message to check if it's spam or not.")
message = st.text_area("Message", height=150)
# if st.button("Predict"):
#     # Step 1: Clean 
#     cleaned = clean_tokenized_sentence(message)
#     # Step 2: Vectorize
#     vectorized = f.transform([cleaned])
#     # Step 3: Predict
#     prediction = mnb.predict(vectorized)[0]
#     result = "spam" if prediction == 1 else "ham"
#     st.write(f"Original Message: {message}")
#     st.write(f"Prediction: {result}")
st.write(message)
text_input = st.text_input("Enter some text")
checkbox = st.checkbox("Check me!")
if checkbox:
    st.write("Checkbox is checked!")
st.divider()
st.write("This is some text", font=("Arial", 16), text_color="red")
st.write(":chart_with_upwards_trend: Some text")
col1, col2 = st.columns(2)
with col1:
    st.header("Column 1")
    st.write("Some data")
with col2:
    st.header("Column 2")
    st.write("Some more data")


