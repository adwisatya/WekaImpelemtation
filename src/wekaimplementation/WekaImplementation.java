/*
 * Main sources: www.programcreek.com/2013/01/a-simple-machine-learning-example-in-java/
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package wekaimplementation;

/**
 *
 * @author adwisatya
 */
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.core.FastVector;
import weka.core.Instances;
 
public class WekaImplementation {
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;
 
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
 
		return inputReader;
	}
 
	public static Evaluation classify(Classifier model,
		Instances trainingSet, Instances testingSet) throws Exception {
		Evaluation evaluation = new Evaluation(trainingSet);
		model.buildClassifier(trainingSet);
		evaluation.evaluateModel(model, testingSet);
 
		return evaluation;
	}
 
	public static double calculateAccuracy(FastVector predictions) {
		double correct = 0;
 
		for (int i = 0; i < predictions.size(); i++) {
			NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
			if (np.predicted() == np.actual()) {
				correct++;
			}
		}
 
		return 100 * correct / predictions.size();
	}
 
	public static Instances[][] crossValidationSplit(Instances data, int numberOfFolds) {
		Instances[][] split = new Instances[2][numberOfFolds];
 
		for (int i = 0; i < numberOfFolds; i++) {
			split[0][i] = data.trainCV(numberOfFolds, i);
			split[1][i] = data.testCV(numberOfFolds, i);
		}
 
		return split;
	}
 
	public static void main(String[] args) throws Exception {
		BufferedReader datafile = readDataFile("weather.nominal.arff");
 
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		//Menampilkan summary dari data
		System.out.println(data.toSummaryString());
		
		// Do 10-split cross validation
		Instances[][] split = crossValidationSplit(data, 10);
 
		// Separate split into training and testing arrays
		Instances[] trainingSplits = split[0];
		Instances[] testingSplits = split[1];
 
		// Use a set of classifiers
		Classifier[] models = { 
				new J48(), //ID3
				new PART(), 
				new DecisionTable(),
				new DecisionStump(),
				new NaiveBayes(), //NaiveBayes
				new IBk(), //k-NN
				new ZeroR() //fullTraining
		};
 
		// Run for each model
		for (int j = 0; j < 1; j++) {
 			FastVector predictions = new FastVector();
 			for (int i = 0; i < trainingSplits.length; i++) {
				Evaluation validation = classify(models[j], trainingSplits[i], testingSplits[i]);
				
				//Menampilkan summary seperti di Weka GUI 
				//System.out.println(validation.toSummaryString()); 
				
				predictions.appendElements(validation.predictions());
			}
			
 			double accuracy = calculateAccuracy(predictions);
			System.out.println("Accuracy of " + models[j].getClass().getSimpleName() + ": "
					+ String.format("%.2f%%", accuracy)
					+ "\n---------------------------------");
		}
 
	}
}