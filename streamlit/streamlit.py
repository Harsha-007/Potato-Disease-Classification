hide_streamlit_style = """
            
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            
            """
st.markdown(hide_streamlit_style, unsafe_allow_html = True)

st.title('Potato Leaf Disease Prediction')

def main() :
    file_uploaded = st.file_uploader('Choose an image...', type = 'jpg')
    if file_uploaded is not None :
        image = Image.open(file_uploaded)
        st.write("Uploaded Image.")
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        st.pyplot(figure)
        result, confidence = predict_class(image)
        st.write('Prediction : {}'.format(result))
        st.write('Confidence : {}%'.format(confidence))

def predict_class(image) :
    with st.spinner('Loading Model...'):
        classifier_model = keras.models.load_model(r'final_model.h5', compile = False)

    shape = ((256,256,3))
    model = keras.Sequential([hub.KerasLayer(classifier_model, input_shape = shape)])     # ye bhi kaam kar raha he
    test_image = image.resize((256, 256))