package com.owldevs.nn4j;

import java.util.function.UnaryOperator;

/**
 * The simple implementation of the multilayer perceptron.
 * Features:
 *     1. Mutation
 *     2. Crossing
 *     3. Education by back propagation method
 *     4. Unlimited Network depth
 *     5. Unlimited Network width
 *     6. Individual width of the each Layer of the Network
 */
public class NeuralNetwork implements Cloneable {
    private double learningRate;
    private Layer[] layers;
    private UnaryOperator<Double> activation = x -> 1 / (1 + Math.exp(-x));
    private UnaryOperator<Double> derivative = x -> x * (1 - x);

    /**
     * Creating new NN with random weights and biases
     *
     * @param learningRate 0..1 The rate of the learning. Makes sense with using backPropagation method for Network
     *                     education. If you do not educate network just set any value here.
     * @param sizes The array which represents quantity of the layers and width of the each layer of the Network.
     *              For example if sizes = [2, 5, 3] will be crated perceptron with 3 layers.
     *              1st layer (inputs) width is 2
     *              2nd layer (hidden layers) width is 5
     *              3rd layer (outputs) width is 3
     */
    public NeuralNetwork(double learningRate, int... sizes) {
        this.learningRate = learningRate;

        layers = new Layer[sizes.length];
        for (int i = 0; i < sizes.length; i++) {
            int nextSize = 0;
            if (i < sizes.length - 1) nextSize = sizes[i + 1];
            layers[i] = new Layer(sizes[i], nextSize);
            for (int j = 0; j < sizes[i]; j++) {
                layers[i].biases[j] = Math.random() * 2.0 - 1.0;
                for (int k = 0; k < nextSize; k++) {
                    layers[i].weights[j][k] = Math.random() * 2.0 - 1.0;
                }
            }
        }
    }

    /**
     * Creating new Network with preconfigured weights and biases.
     *
     * @param learningRate 0..1 The rate of the learning. Makes sense with using backPropagation method for Network
     *                     education. If you do not educate network just set any value here.
     * @param layers The array of the layers of the Network
     */
    public NeuralNetwork (double learningRate, Layer[] layers) {
        this.learningRate = learningRate;
        this.layers = layers;
    }

    /**
     * Push signals to input of Network and receive Network calculation result from the Network output
     * @param inputs Array of params. The size of the array has to be more or equal
     *               the size of the 1st Layer of the Network.
     * @return Network output values. Size equals to the size og the last layer of the Network
     */
    public double[] feedForward(double[] inputs) {
        System.arraycopy(inputs, 0, layers[0].neurons, 0, inputs.length);
        for (int i = 1; i < layers.length; i++) {
            Layer l = layers[i - 1];
            Layer l1 = layers[i];
            for (int j = 0; j < l1.size; j++) {
                l1.neurons[j] = 0;
                for (int k = 0; k < l.size; k++) {
                    l1.neurons[j] += l.neurons[k] * l.weights[k][j];
                }
                l1.neurons[j] += l1.biases[j];
                l1.neurons[j] = (activation.apply(l1.neurons[j])) * 2 - 1;
            }
        }
        return layers[layers.length - 1].neurons;
    }

    /**
     * Educate Network by back propagation method.
     * This method has to be called right after calling feedForward.
     * Targets param shows the Network which result it has to associate with
     * input values which came in feedForward()
     *
     * @param targets The proper values of output which should correspond to las call of feedForward()
     */
    public void backPropagation(double[] targets) {
        double[] errors = new double[layers[layers.length - 1].size];
        for (int i = 0; i < layers[layers.length - 1].size; i++) {
            errors[i] = targets[i] - layers[layers.length - 1].neurons[i];
        }
        for (int k = layers.length - 2; k >= 0; k--) {
            Layer l = layers[k];
            Layer l1 = layers[k + 1];
            double[] errorsNext = new double[l.size];
            double[] gradients = new double[l1.size];
            for (int i = 0; i < l1.size; i++) {
                gradients[i] = errors[i] * derivative.apply(layers[k + 1].neurons[i]);
                gradients[i] *= learningRate;
            }
            double[][] deltas = new double[l1.size][l.size];
            for (int i = 0; i < l1.size; i++) {
                for (int j = 0; j < l.size; j++) {
                    deltas[i][j] = gradients[i] * l.neurons[j];
                }
            }
            for (int i = 0; i < l.size; i++) {
                errorsNext[i] = 0;
                for (int j = 0; j < l1.size; j++) {
                    errorsNext[i] += l.weights[i][j] * errors[j];
                }
            }
            errors = new double[l.size];
            System.arraycopy(errorsNext, 0, errors, 0, l.size);
            double[][] weightsNew = new double[l.weights.length][l.weights[0].length];
            for (int i = 0; i < l1.size; i++) {
                for (int j = 0; j < l.size; j++) {
                    weightsNew[j][i] = l.weights[j][i] + deltas[i][j];
                }
            }
            l.weights = weightsNew;
            for (int i = 0; i < l1.size; i++) {
                l1.biases[i] += gradients[i];
            }
        }
    }

    /**
     * Mutate the current instance with specified chance and rate
     * @param rate Rate set the maximal possible delta on which the value of
     *             Weights and Biases can be modified to both sides -+.
     *             The range of values of Weights and Biases is from -1 to 1.
     *             Thar is why the limit of range is 2, but we recommend
     *             do use rate value between 0 and 0.2. It makes better results.
     * @param chance 0..1 where 1 means 100%
     */
    public void mutate(double rate, double chance) {
        for (int i = 0; i < layers.length; i++){
            for(int j = 0; j < layers[i].weights.length; j++) {
                for(int g = 0; g < layers[i].weights[j].length; g++) {
                    if (Math.random() < chance) {
                        layers[i].weights[j][g] += (Math.random() - 0.5d) * rate;

                        if(layers[i].weights[j][g] > 1) layers[i].weights[j][g] = 1;
                        if(layers[i].weights[j][g] < -1) layers[i].weights[j][g] = -1;
                    }
                }
            }

            for(int j = 0; j < layers[i].biases.length; j++) {

                    if (Math.random() < chance) {
                        layers[i].biases[j] += (Math.random() - 0.5d) * rate;

                        if(layers[i].biases[j] > 1) layers[i].biases[j] = 1;
                        if(layers[i].biases[j] < -1) layers[i].biases[j] = -1;
                    }

            }
        }
    }

    /**
     * Cross the current Network with the target.
     *
     * @param targetNetwork The target instance of NeuralNetwork
     */
    public void crossWith(NeuralNetwork targetNetwork)
    {
        Layer[] layer = targetNetwork.layers.clone();
        for (int i = 0; i < layers.length; i++){
            for(int j = 0; j < layers[i].weights.length; j++) {
                for(int g = 0; g < layers[i].weights[j].length; g++) {
                    if (Math.random() < 0.5) {
                        layers[i].weights[j][g] = layer[i].weights[j][g];
                    }
                }
            }
            for(int j = 0; j < layers[i].biases.length; j++) {
                if (Math.random() < 0.5) {
                    layers[i].biases[j] = layer[i].biases[j];
                }
            }
        }
    }

    /**
     * Clone of the current NeuralNetwork instance
     *
     * @return new instance of the NeuralNetwork
     */
    @Override
    public NeuralNetwork clone() {
        Layer[] newLayers = new Layer[layers.length];
        for (int i = 0; i < layers.length; i++) {
            int nextSize = 0;
            if (i < layers.length - 1) nextSize = layers[i + 1].size;
            newLayers[i] = new Layer(layers[i].size, nextSize);
            for (int j = 0; j < layers[i].size; j++) {
                newLayers[i].biases[j] = layers[i].biases[j];
                for (int k = 0; k < nextSize; k++) {
                    newLayers[i].weights[j][k] = layers[i].weights[j][k];
                }
            }
        }

        return new NeuralNetwork(learningRate, newLayers);
    }

    /**
     * Override default Activation Function
     * @param activation Activation Function
     * @return this
     */
    public NeuralNetwork setActivationFunction(UnaryOperator<Double> activation) {
        this.activation = activation;
        return this;
    }

    /**
     * Override default Derivative Function
     * @param derivative Derivative Function
     * @return this
     */
    public NeuralNetwork setDerivativeFunction(UnaryOperator<Double> derivative) {
        this.derivative = derivative;
        return this;
    }

}