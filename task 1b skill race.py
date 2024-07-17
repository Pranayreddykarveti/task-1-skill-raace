#!/usr/bin/env python
# coding: utf-8

# In[1]:


import java.util.ArrayList;
import java.util.List;
import net.sf.javaml.classification.Classifier;
import net.sf.javaml.classification.bayes.NaiveBayesClassifier;
import net.sf.javaml.classification.evaluation.CrossValidation;
import net.sf.javaml.classification.evaluation.Precision;
import net.sf.javaml.classification.evaluation.Recall;
import net.sf.javaml.classification.evaluation.EvaluationMetrics;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.core.SparseInstance;
import net.sf.javaml.tools.data.FileHandler;

public class NaiveBayesTextClassifier {

    public static void main(String[] args) throws Exception {
        // Load dataset
        Dataset data = new DefaultDataset();
        
        // Add documents and their labels
        // Format: new SparseInstance({feature1, feature2, ...}, label)
        data.add(new SparseInstance(new double[]{1, 0, 1, 0, 1}, "spam"));
        data.add(new SparseInstance(new double[]{0, 1, 0, 1, 0}, "not spam"));
        data.add(new SparseInstance(new double[]{1, 1, 1, 0, 0}, "spam"));
        data.add(new SparseInstance(new double[]{0, 0, 0, 1, 1}, "not spam"));
        // Add more documents as needed

        // Create Naive Bayes classifier
        Classifier nb = new NaiveBayesClassifier(false, true, true);

        // Train the classifier
        nb.buildClassifier(data);

        // Evaluate the classifier using cross-validation
        CrossValidation cv = new CrossValidation(nb);
        Dataset[] split = cv.split(data, 10);
        Dataset trainingData = split[0];
        Dataset testData = split[1];

        // Train on training data
        nb.buildClassifier(trainingData);

        // Test on test data
        List<String> predictions = new ArrayList<>();
        List<String> actuals = new ArrayList<>();
        for (Instance instance : testData) {
            Object predictedClassValue = nb.classify(instance);
            Object realClassValue = instance.classValue();
            predictions.add(predictedClassValue.toString());
            actuals.add(realClassValue.toString());
        }

        // Calculate accuracy, precision, and recall
        EvaluationMetrics evaluationMetrics = new EvaluationMetrics();
        double accuracy = evaluationMetrics.calculateAccuracy(predictions, actuals);
        double precision = evaluationMetrics.calculatePrecision(predictions, actuals);
        double recall = evaluationMetrics.calculateRecall(predictions, actuals);

        System.out.println("Accuracy: " + accuracy);
        System.out.println("Precision: " + precision);
        System.out.println("Recall: " + recall);
    }
}


# In[ ]:




