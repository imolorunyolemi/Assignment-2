# Assignment-2 (FIRST PART) (Imoleayo Moses Olorunyolemi)


***Part1:Visualize the Data***

To set the title and label for 'x' and 'y' axis for the scatter plot, I copied the following codes from the 'data visualization notebook' and pasted it in the available coding box for the recent assignment (Part 1):

**plt.title('Scatter Plot with Seaborn')  # Title of the plot**

**plt.xlabel('X-axis')  # Label for x-axis**

**plt.ylabel('Y-axis')  # Label for y-axis**

I then went ahead to edit the statements in the brackets () to the required title as instructed. Hence, the complete edited codes below:

**plt.figure(figsize=(10, 6))**

**plt.scatter(wine_df['alcohol'], wine_df['malic_acid'], c=wine_df['target'], cmap='coolwarm', s=30)**

**plt.title('Wine Quality Dataset')  # Title of the plot**

**plt.xlabel('Alcohol')  # Label for x-axis**

**plt.ylabel('Malic Acid')  # Label for y-axis**

**plt.colorbar(ticks=np.unique(y), label='Quality')**

**plt.show()**

Finally, I ran the edited code to give the resultant figure (chart) as shown below:

![Assignment 2 Part 1](https://github.com/user-attachments/assets/79473d9e-4be0-4635-8005-41c21444b58c)


***Part2: Split the Data***

To split the dataset into training and testing sets, I copied the example code in the 'Load DataSet cell' as shown below:

**iris = datasets.load_iris()**

**X, y = iris.data, iris.target**

**X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)**

I then pasted the copied codes in the available space, and edited the dataset from 'iris' to 'wine' to have the new codes as shown below:

**wine = datasets.load_wine()**

**X, y = wine.data, wine.target**

**X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)**

Finally, I ran the code.

![Assignment 2 Part 2](https://github.com/user-attachments/assets/87a35f2f-c141-4089-a5ad-b7936ac075d1)


***Part 3 : create a Decision Tree and a Logistic Regression model to classify the wine quality. We will print the accuracy for each model. The code for SVM will serve as an example, and you should complete the code for Decision Tree and Logistic Regression accordingly.***

To create a Decision Tree Model to classify the wine quality, and print the accuracy for the model, I first copied the codes for 'training the decision tree model' from the example on decision tree given in class and pasted in the available space for 'Train the Decision Tree model', as shown below:

**tree = DecisionTreeClassifier() # Train the Decision Tree**

**tree.fit(X_train, y_train)  # Train the Decision Tree**

Then I went on to copy the 'Make predictions' code from the same example as above and pasted in the available space for 'Make predictions' as shown below:

**y_pred = tree.predict(X_test) # Make predictions**

Furthermore, I copied the code for 'evaluating the model' from the SVM code example given under this same part 3 of the homework. I then pasted this copied codes in the available space for 'Evaluate the model' as shown below:

**#Make predictions and calculate accuracy**

**accuracy = accuracy_score(y_test , y_pred)**

**print(f"Accuracy of the SVM model: {accuracy:.2f}")**

I went on to edit the 'SVM' in the 'print' code to 'Decision Tree', to have a new code shown below:

**#Make predictions and calculate accuracy**

**accuracy = accuracy_score(y_test , y_pred)**

**print(f"Accuracy of the Decision Tree model: {accuracy:.2f}")**

I also tried visualizing the decision tree by copying the codes for the 'Visualization of decision tree' and editing the 'iris' to 'wine'. This is also depicted below:

**#Visualization of decision tree**

**dot_data = export_graphviz(tree, out_file=None, feature_names=wine.feature_names, class_names=wine.target_names, filled=True)**

**graph = graphviz.Source(dot_data)**

**graph.render("wine_decision_tree", format="png")**

**graph**

Finally, I ran the code. I was able to visualize the decision tree charts as well as print the 'accuracy for the decision tree model' as 0.96

![Assignment 2 Part 3ai (decision tree)](https://github.com/user-attachments/assets/d7588ee8-8d16-4399-a403-ee4342049caa)

![Assignment 2 Part 3aii (decision tree)](https://github.com/user-attachments/assets/09204657-6dc1-4db2-ad37-96920311b296)

![Assignment 2 Part 3aiii (decision tree)](https://github.com/user-attachments/assets/687dd861-8708-4081-84ea-1877cd7ddbeb)

![Assignment 2 Part 3aiv (decision tree)](https://github.com/user-attachments/assets/907221ab-b2da-48e5-a9a2-a7757e28cdb2)


To create a Logistic Regression Model to classify the wine quality, and print the accuracy for the model, I first copied the codes for 'training the Logistic Regression model' from the example on Logistic Regression given in class and pasted in the available space for 'Train the Logistic Regression model', as shown below:

**log_regressor = LogisticRegression(max_iter=200) #Train the Logistic Regression**

**log_regressor.fit(X_train[:, :2], y_train) #Train the Logistic Regression**

I also tried visualizing the Logistic Regression Decision Boundary by copying and pasting the codes for the 'Visualization'. This is also depicted below:

**#Visualization**

**plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis')**

**plt.title("Logistic Regression Decision Boundary")**

**plt.show()**

Then I went on to copy the 'Make predictions' code from the same example as above and pasted in the available space for 'Make predictions' as shown below:

**y_pred = log_regressor.predict(X_test[:, :2]) # Make predictions**

Furthermore, I copied the code for 'evaluating the model' from the SVM code example given under this same part 3 of the homework. I then pasted this copied codes in the available space for 'Evaluate the model' as shown below:

**#Make predictions and calculate accuracy**

**accuracy = accuracy_score(y_test , y_pred)**

**print(f"Accuracy of the SVM model: {accuracy:.2f}")**

I went on to edit the 'SVM' in the 'print' code to 'Logistic Regression', to have a new code shown below:

**#Make predictions and calculate accuracy**

**accuracy = accuracy_score(y_test , y_pred)**

**print(f"Accuracy of the Logistic Regression model: {accuracy:.2f}")**

Finally, I ran the code. I was able to visualize the Logistic Regression Decision Boundary scatter plot as well as print the 'accuracy for the Logistic Regression model' as 0.74

![Assignment 2 Part 3b (Logistic Regression)](https://github.com/user-attachments/assets/cc060f7b-3869-4e05-a067-5e313b0565dc)

![Assignment 2 Part 3b (Logistic Regression Decision Boundary)](https://github.com/user-attachments/assets/fa94b8f0-16a3-46cf-be4c-060756ca3080)


# Assignment-2 (SECOND PART) (Imoleayo Moses Olorunyolemi)

**Part1 : In this section, we want to classify the MNIST dataset using Random Forest, Logistic Regression, and Neural Networks. We will print the accuracy for each model. The code for Random Forest will serve as an example, and you should complete the code for Logistic Regression and Neural Networks accordingly.**

To classify the MNIST dataset using Logistic Regression, I copied the following codes from the Logistic Regression example in the 'machine learning notebook' from previous class and pasted it in the available coding box for the recent assignment (Part 1a):

**log_regressor = LogisticRegression(max_iter=200) #Train the Logistic Regression**

**log_regressor.fit(X_train[:, :2], y_train) #Train the Logistic Regression**

I then went ahead to edit the codes to fit the required name and title for this particular data as instructed. Hence, the complete edited codes below:

**logistic_classifier = LogisticRegression(max_iter=200) #Train the Logistic Regression**

**logistic_classifier.fit(X_train, y_train) #Train the Logistic Regression**

I also copied the print accuracy prediction code from the "Random Forest" example and edited to fit the data for "Logistic Regression" as shown below:

**print(accuracy_logistic)**

Finally, I ran the code to get the printed 'accuracy for the Logistic Regression' as 0.919.

![Assignment 2 Part 4ai (Logistic Regression)](https://github.com/user-attachments/assets/2c5695c7-c440-48b5-ae0d-07def16009ac)


To classify the MNIST dataset using Neural Networks, I copied the following codes from the MLP (Neural Network) example in the 'machine learning notebook' from previous class and pasted it in the available coding box for the recent assignment (Part 1b):

**mlp = MLPClassifier(max_iter=1000) # Train the MLP**

**mlp.fit(X_train, y_train) # Train the MLP**

I also copied the print accuracy prediction code from the "Random Forest" example and edited to fit the data for "MLP (Neural Network)" as shown below:

**print(accuracy_mlp)**

Finally, I ran the code and got the printed 'accuracy for the MLP (Neural Network)' as 0.978.

![Assignment 2 Part 4aii (MLP Neural Network)](https://github.com/user-attachments/assets/2dbec8d6-2e4e-4d15-94b6-d8f68b4a21b0)

Note: All of my printed accuracy correlated with the percentage printed accuracy already calculated in "step 6" in the homework.

![Assignment 2 Part 4aiii (Print Accuracies)](https://github.com/user-attachments/assets/5c188377-d892-4fde-b000-51782f0f4ff4)


**Part2: take a photo of yourself, upload it to the jupyter notebook and detect your face using Yolo model**

To detect my face from my picture using Yolo method, I initially uploaded my picture to the jupyter notebook by dragging my image to the file folder in the notebook.

I then copied the codes for one of the "Object Detection" example (Game of thrones) given in class and pasted it in the available space for my code as shown below:

**img_path = '/content/game of thrones.jpeg'  # path to your image**

**image = Image.open(img_path)**


**#Resize the image**

**resized_image = image.resize((800, 600))**


**#Convert the resized image to a format compatible with the model (e.g., NumPy array)**

**#If your model requires a specific input format, you may need to adjust the image further.**

**img_array = np.array(resized_image)**


**#Perform inference**

**results = model(img_array)**


**#Show results**

**results.show()  # This will display the image with detections**


**#If you want to visualize the resized image**

**plt.imshow(resized_image)**

**plt.axis('off')**

**plt.title('Resized Image')**

**plt.show()**

I then copied the 'path' for my uploaded picture and used it to replace the path in the "Object Detection" code that was previously copied. Hence, my new code looks like this:

**img_path = '/content/Imole.jpg'  # path to your image**

**image = Image.open(img_path)**


**#Resize the image**

**resized_image = image.resize((800, 600))**


**#Convert the resized image to a format compatible with the model (e.g., NumPy array)**

**#If your model requires a specific input format, you may need to adjust the image further.**

**img_array = np.array(resized_image)**


**#Perform inference**

**results = model(img_array)**


**#Show results**

**results.show()  # This will display the image with detections**


**#If you want to visualize the resized image**

**plt.imshow(resized_image)**

**plt.axis('off')**

**plt.title('Resized Image')**

**plt.show()**

Finally, I ran the code and detected my face as a person's face with an accuracy of 0.89.

![Assignment 2 Part 5](https://github.com/user-attachments/assets/da072cfc-91ba-4d3b-abf1-7fa0c623b8e8)
