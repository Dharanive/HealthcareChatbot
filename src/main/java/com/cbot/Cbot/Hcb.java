package com.cbot.Cbot;

import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class Hcb {

    private static final String TRAIN_FILE = "Data/trainingSet.csv";
    private static final String TEST_FILE = "Data/testingSet.csv";
    private static final String PRECAUTIONS_FILE = "MasterData/symptom_precaution.csv";
    private static final String SEVERITY_FILE = "MasterData/Symptom_severity.csv";
    private static final String DESCRIPTION_FILE = "MasterData/symptom_Description.csv";

    public static void main(String[] args) throws Exception {
        Scanner scanner = new Scanner(System.in);

        

        // Load dataset from CSV file
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("Data/dataset.csv"));
        Instances data = loader.getDataSet();

        // Set class attribute
        data.setClassIndex(data.numAttributes() - 1);

        // Split dataset into training and testing sets
        Instances trainData = new Instances(data, 0, (int) (data.numInstances() * 0.8)); // 80% for training
        Instances testData = new Instances(data, (int) (data.numInstances() * 0.8), (int) (data.numInstances() * 0.2)); // 20% for testing

        // Save training and testing datasets to CSV files
        saveInstancesToCSV(trainData, TRAIN_FILE);
        saveInstancesToCSV(testData, TEST_FILE);

        // Build a Random Forest model using training data
        Classifier classifier = new RandomForest();
        classifier.buildClassifier(trainData);
        
     // Greeting message and ask for user's name
        System.out.println("Welcome to the Healthcare Chatbot!");
        System.out.print("May I know your name? ");
        String userName = scanner.nextLine().trim();

        // Interactive symptom input and prediction
        predictDisease(classifier, testData, scanner);

        // Closing message with user's name
        System.out.println("Thank you, " + userName + "! Have a great day.");

        scanner.close();
    }

    private static void saveInstancesToCSV(Instances data, String filename) throws IOException {
        CSVSaver saver = new CSVSaver();
        saver.setInstances(data);
        saver.setFile(new File(filename));
        saver.writeBatch();
    }

    private static void predictDisease(Classifier classifier, Instances testData, Scanner scanner) throws Exception {
        boolean predicting = true;

        while (predicting) {
            Map<String, Double> symptomsMap = new HashMap<>();
            String previousSymptom = null;

            while (true) {
                if (previousSymptom == null) {
                    // Ask for the first symptom
                    System.out.println("Enter symptom name (or 'done' to finish):");
                    previousSymptom = scanner.nextLine().trim();

                    if (previousSymptom.equalsIgnoreCase("done")) {
                        predicting = false;
                        continue; // Exit inner loop
                    }

                    // Automatically set the value to 1 for the first symptom
                    symptomsMap.put(previousSymptom, 1.0); // Get value for the first symptom
                } else {
                    // Determine the next symptom based on the previous one
                    String nextSymptom = getNextSymptom(previousSymptom, testData);
                    if (nextSymptom == null) {
                        // No more symptoms to ask, trigger disease prediction
                        break; // Exit inner loop
                    }

                    // Ask for symptom value (yes or no)
                    System.out.print("Do you have " + nextSymptom + " ? (yes or no): ");
                    double symptomValue = getSymptomValueFromUser(scanner);
                    symptomsMap.put(nextSymptom, symptomValue);

                    // Update the previousSymptom to the current one
                    previousSymptom = nextSymptom;
                    
                    if (symptomsMap.size() >= 8) {
                        break;
                    }
                }
            }

            if (!symptomsMap.isEmpty()) {
                // Count the number of 'yes' responses in symptomsMap
                long numberOfYesResponses = symptomsMap.values().stream().filter(value -> value == 1.0).count();

                if (numberOfYesResponses <= 2) {
                    // Insufficient symptoms provided for prediction
                    System.out.println("Sorry, the symptoms provided are not sufficient to predict a disease.");
                } else {
                    DenseInstance newInstance = new DenseInstance(testData.numAttributes());
                    newInstance.setDataset(testData);

                    // Set symptom values in the instance
                    for (int i = 0; i < testData.numAttributes() - 1; i++) {
                        String attributeName = testData.attribute(i).name();
                        double symptomValue = symptomsMap.getOrDefault(attributeName, 0.0);
                        newInstance.setValue(i, symptomValue);
                    }

                    // Get the predicted class label
                    double predictedClass = classifier.classifyInstance(newInstance);
                    String predictedDisease = testData.classAttribute().value((int) predictedClass);

                    // Output predicted disease
                    System.out.println("Predicted Disease: " + predictedDisease);

                    // Display additional information (precautions, severity, description)
                    displayAdditionalInfo(predictedDisease);
                }

                // Ask for another prediction
                System.out.println("Do you want to predict another disease? (yes or no): ");
                String continuePrediction = scanner.nextLine().trim().toLowerCase();
                if (continuePrediction.equals("no")) {
                    predicting = false;
                }
            }
        }
    }

    private static double getSymptomValueFromUser(Scanner scanner) {
        String userInput = scanner.nextLine().trim().toLowerCase();
        return (userInput.equals("yes")) ? 1.0 : 0.0;
    }


    private static String getNextSymptom(String previousSymptom, Instances data) {
        // Determine the next symptom based on the previous symptom
        for (int i = 0; i < data.numAttributes() - 1; i++) {
            String attributeName = data.attribute(i).name();
            if (attributeName.equalsIgnoreCase(previousSymptom)) {
                // Return the next attribute (symptom) name
                return data.attribute(i + 1).name();
            }
        }
        return null; // No more symptoms to ask
    }

    private static void displayAdditionalInfo(String predictedDisease) {
        System.out.println("Additional Information:");
        displayPrecautions(predictedDisease);
        displaySeverityMessage(predictedDisease);
        displayDescription(predictedDisease);
        System.out.println();
    }

    private static void displayPrecautions(String predictedDisease) {
        System.out.println("Precautions:");

        try (Scanner scanner = new Scanner(new FileReader(PRECAUTIONS_FILE))) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                String[] parts = line.split(",");
                if (parts.length >= 2 && parts[0].equalsIgnoreCase(predictedDisease)) {
                    // Extract precautions from the line (excluding the disease name)
                    String[] precautions = Arrays.copyOfRange(parts, 1, parts.length);
                    for (String precaution : precautions) {
                        System.out.println("- " + precaution.trim());
                    }
                    break; // Stop after displaying precautions for the predicted disease
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void displaySeverityMessage(String predictedDisease) {
        System.out.println("Severity:");
        int severityValue = getSeverity(predictedDisease);
        if (severityValue > 3) {
            System.out.println("- Severity level is high. You should take consultation from a doctor.");
        } else {
            System.out.println("- Severity level is moderate. It might not be that bad, but you should take precautions.");
        }
    }

    private static void displayDescription(String predictedDisease) {
        System.out.println("Description:");
        try (Scanner scanner = new Scanner(new FileReader(DESCRIPTION_FILE))) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                String[] parts = line.split(",");
                if (parts.length >= 2 && parts[0].equalsIgnoreCase(predictedDisease)) {
                    // Build the description excluding the disease name
                    StringBuilder description = new StringBuilder();
                    for (int i = 1; i < parts.length; i++) {
                        description.append(parts[i]); // Append each part of the description
                        if (i < parts.length - 1) {
                            description.append(", "); // Add comma between parts
                        }
                    }
                    System.out.println("- " + description.toString());
                    break;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static int getSeverity(String predictedDisease) {
        try (Scanner scanner = new Scanner(new FileReader(SEVERITY_FILE))) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                String[] parts = line.split(",");
                if (parts.length >= 2 && parts[0].equalsIgnoreCase(predictedDisease)) {
                    return Integer.parseInt(parts[1]);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return -1; // Return default severity value if not found
    }
}
