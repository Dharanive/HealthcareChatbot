package com.cbot.Cbot;

import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class App extends JFrame implements ActionListener {

    private static final String TRAIN_FILE = "Data/trainingSet.csv";
    private static final String TEST_FILE = "Data/testingSet.csv";
    private static final String PRECAUTIONS_FILE = "MasterData/symptom_precaution.csv";
    private static final String SEVERITY_FILE = "MasterData/Symptom_severity.csv";
    private static final String DESCRIPTION_FILE = "MasterData/symptom_Description.csv";

    private JTextField nameField;
    private JTextField symptomField;
    private JTextArea chatArea;
    private JButton startButton;
    private JButton submitButton;
    private Instances testData;
    private Classifier classifier;

    private String previousSymptom;
    private boolean predicting;

    public App() {
        super("Healthcare Chatbot");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(600, 400);
        setLayout(new BorderLayout());

        initComponents();

        // Load dataset and build classifier
        try {
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File("Data/dataset.csv"));
            Instances data = loader.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

            this.testData = data;
            this.classifier = new RandomForest();
            this.classifier.buildClassifier(data);

        } catch (Exception e) {
            e.printStackTrace();
            JOptionPane.showMessageDialog(this, "Error initializing the chatbot. Please try again later.");
        }
    }

    private void initComponents() {
        JPanel topPanel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        JLabel nameLabel = new JLabel("Your Name:");
        nameField = new JTextField(15);
        startButton = new JButton("Start Chat");
        startButton.addActionListener(this);

        topPanel.add(nameLabel);
        topPanel.add(nameField);
        topPanel.add(startButton);

        add(topPanel, BorderLayout.NORTH);

        chatArea = new JTextArea();
        chatArea.setEditable(false);
        JScrollPane scrollPane = new JScrollPane(chatArea);
        add(scrollPane, BorderLayout.CENTER);

        JPanel inputPanel = new JPanel(new BorderLayout());
        JLabel symptomLabel = new JLabel("Enter symptom :");
        symptomField = new JTextField(20);
        submitButton = new JButton("Submit");
        submitButton.addActionListener(this);

        inputPanel.add(symptomLabel, BorderLayout.WEST);
        inputPanel.add(symptomField, BorderLayout.CENTER);
        inputPanel.add(submitButton, BorderLayout.EAST);

        add(inputPanel, BorderLayout.SOUTH);
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == startButton) {
            // Start chat button clicked
            String userName = nameField.getText().trim();
            if (userName.isEmpty()) {
                JOptionPane.showMessageDialog(this, "Please enter your name.");
            } else {
                chatArea.append("Welcome to the Healthcare Chatbot, " + userName + "!\n");
                nameField.setEditable(false);
                startButton.setEnabled(false);
                predicting = true;
                previousSymptom = null;
            }
        } else if (e.getSource() == submitButton && predicting) {
            // Submit button clicked during chat session
            String userInput = symptomField.getText().trim();
            
            if (previousSymptom == null) {
                // Expecting a symptom
                previousSymptom = userInput;
                chatArea.append("Do you have " + previousSymptom + "? (yes or no)\n");
            } else {
                // Expecting the response ("yes" or "no") to the symptom
                String response = userInput.toLowerCase();
                if (response.equals("yes") || response.equals("no")) {
                    try {
                        // Process the response
                        double value = response.equals("yes") ? 1.0 : 0.0;
                        predictDisease(previousSymptom, value);
                    } catch (Exception ex) {
                        ex.printStackTrace();
                        JOptionPane.showMessageDialog(this, "Error predicting disease. Please try again.");
                    }
                } else {
                    JOptionPane.showMessageDialog(this, "Please enter 'yes' or 'no'.");
                }
                previousSymptom = null; // Reset to accept the next symptom
            }
            symptomField.setText("");
        }
    }

    private void predictDisease(String symptom, double value) throws Exception {
        Map<String, Double> symptomsMap = new HashMap<>();
        symptomsMap.put(symptom, value);

        // Display the first symptom and its value in the chatArea
        chatArea.append("Symptom: " + symptom + ", Response: " + (value == 1.0 ? "Yes" : "No") + "\n");

        while (symptomsMap.size() < 8) {
            String nextSymptom = getNextSymptom(symptom);
            if (nextSymptom == null) {
                break; // No more symptoms to ask
            }
            String response = JOptionPane.showInputDialog("Do you have " + nextSymptom + "? (yes or no)");
            double responseValue = (response != null && response.equalsIgnoreCase("yes")) ? 1.0 : 0.0;
            symptomsMap.put(nextSymptom, responseValue);
            symptom = nextSymptom;

            // Display the next symptom and its value in the chatArea
            chatArea.append("Symptom: " + nextSymptom + ", Response: " + (responseValue == 1.0 ? "Yes" : "No") + "\n");
        }

        if (!symptomsMap.isEmpty()) {
            // Count the number of 'yes' responses in symptomsMap
            long numberOfYesResponses = symptomsMap.values().stream().filter(values -> values == 1.0).count();

            if (numberOfYesResponses < 3) {
                chatArea.append("Sorry, the symptoms provided are not sufficient to predict a disease.\n");
            } else {
                DenseInstance newInstance = new DenseInstance(testData.numAttributes());
                newInstance.setDataset(testData);

                for (int i = 0; i < testData.numAttributes() - 1; i++) {
                    String attributeName = testData.attribute(i).name();
                    double symptomValue = symptomsMap.getOrDefault(attributeName, 0.0);
                    newInstance.setValue(i, symptomValue);
                }

                double predictedClass = classifier.classifyInstance(newInstance);
                String predictedDisease = testData.classAttribute().value((int) predictedClass);

                chatArea.append("Predicted Disease: " + predictedDisease + "\n");
                displayAdditionalInfo(predictedDisease);
                
                
            }
        }
    }





    private String getNextSymptom(String previousSymptom) {
        for (int i = 0; i < testData.numAttributes() - 1; i++) {
            String attributeName = testData.attribute(i).name();
            if (attributeName.equalsIgnoreCase(previousSymptom)) {
                return testData.attribute(i + 1).name();
            }
        }
        return null;
    }

    


    private void displayAdditionalInfo(String predictedDisease) {
        displayPrecautions(predictedDisease);
        displaySeverityMessage(predictedDisease);
        displayDescription(predictedDisease);
    }

    private void displayPrecautions(String predictedDisease) {
        StringBuilder precautionsText = new StringBuilder("Precautions:\n");

        try (Scanner scanner = new Scanner(new FileReader(PRECAUTIONS_FILE))) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                String[] parts = line.split(",");
                if (parts.length >= 2 && parts[0].equalsIgnoreCase(predictedDisease)) {
                    // Extract precautions from the line (excluding the disease name)
                    for (int i = 1; i < parts.length; i++) {
                        precautionsText.append("- ").append(parts[i].trim()).append("\n");
                    }
                    break; // Stop after displaying precautions for the predicted disease
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
            precautionsText.append("Error fetching precautions.");
        }

        JOptionPane.showMessageDialog(this, precautionsText.toString(), "Precautions for " + predictedDisease,
                JOptionPane.INFORMATION_MESSAGE);
    }

    private void displaySeverityMessage(String predictedDisease) {
        int severityValue = getSeverity(predictedDisease);
        String severityMessage;
        if (severityValue > 3) {
            severityMessage = "Severity level is high. You should take consultation from a doctor.";
        } else {
            severityMessage = "Severity level is moderate. It might not be that bad, but you should take precautions.";
        }

        JOptionPane.showMessageDialog(this, severityMessage, "Severity for " + predictedDisease,
                JOptionPane.INFORMATION_MESSAGE);
    }

    private void displayDescription(String predictedDisease) {
        StringBuilder descriptionText = new StringBuilder("Description:\n");

        try (Scanner scanner = new Scanner(new FileReader(DESCRIPTION_FILE))) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                String[] parts = line.split(",");
                if (parts.length >= 2 && parts[0].equalsIgnoreCase(predictedDisease)) {
                    // Build the description excluding the disease name
                    for (int i = 1; i < parts.length; i++) {
                        descriptionText.append("- ").append(parts[i].trim());
                        if (i < parts.length - 1) {
                            descriptionText.append(", ");
                        }
                    }
                    break;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
            descriptionText.append("Error fetching description.");
        }

        JOptionPane.showMessageDialog(this, descriptionText.toString(), "Description for " + predictedDisease,
                JOptionPane.INFORMATION_MESSAGE);
    }

    private int getSeverity(String predictedDisease) {
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

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            App chatbotUI = new App();
            chatbotUI.setVisible(true);
        });
    }
}
