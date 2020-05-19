package com.owldevs.nn4j;

import java.util.function.UnaryOperator;

/**
 *
 */
public class NeuralNetwork implements Cloneable {

    private double learningRate;
    private Layer[] layers;
    private UnaryOperator<Double> activation;
    private UnaryOperator<Double> derivative;

    /**
     * Creating new NN with random weights and biases
     * @param learningRate
     * @param sizes
     */
    public NeuralNetwork(double learningRate, int... sizes) {
        this.learningRate = learningRate;
        this.activation = xx -> 1 / (1 + Math.exp(-xx));
        this.derivative = xx -> xx * (1 - xx);

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
     * Creating new NN with preconfigured weights and biases
     * @param learningRate
     * @param layers
     */
    public NeuralNetwork (double learningRate, Layer[] layers) {
        this.learningRate = learningRate;
        this.activation = x -> 1 / (1 + Math.exp(-x));
        this.derivative = y -> y * (1 - y);
        this.layers = layers;
    }

    /**
     *
     * @param inputs
     * @return
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
     *
     * @param targets
     */
    public void backpropagation(double[] targets) {
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
     *
     * @param rate
     * @param chance
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
     *
     * @param nn
     */
    public void crossover(NeuralNetwork nn)
    {
        Layer[] lr = nn.layers.clone();
        for (int i = 0; i < layers.length; i++){
            for(int j = 0; j < layers[i].weights.length; j++) {
                for(int g = 0; g < layers[i].weights[j].length; g++) {
                    if (Math.random() < 0.5) {
                        layers[i].weights[j][g] = lr[i].weights[j][g];
                    }
                }
            }

            for(int j = 0; j < layers[i].biases.length; j++) {
                if (Math.random() < 0.5) {
                    layers[i].biases[j] = lr[i].biases[j];
                }
            }
        }
    }

    /**
     *
     * @return
     */
    @Override
    public  NeuralNetwork clone(){
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


}