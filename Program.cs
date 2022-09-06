using System;
using System.Collections.Generic;
using System.IO;
using CNTK;
using System.Linq;

namespace szak_rendszerek_beadando
{
    class NeuralNetwork
    {
        const int inputSize = 4; //bemeneti oszlopok a táblázatból
        const int hiddenNeuronCount = 3; // neurális háló rejtett rétegek neuronjainak száma
        const int outputSize = 1; // kimenet

        readonly Variable x; // bemeneti réteg
        readonly Function y; // kimeneti réteg
        

        public NeuralNetwork()
        {
            // Build graph
            x = Variable.InputVariable(new int[] { inputSize, 1 }, DataType.Float); // bemeneti réteg
            Parameter w1 = new Parameter(new int[] { hiddenNeuronCount, inputSize }, DataType.Float, CNTKLib.GlorotNormalInitializer()); // rejtett réteg súly értéke
            Parameter b = new Parameter(new int[] { hiddenNeuronCount, 1 }, DataType.Float, CNTKLib.GlorotNormalInitializer());           // rejtett réteg
            Parameter w2 = new Parameter(new int[] { outputSize, hiddenNeuronCount }, DataType.Float, CNTKLib.GlorotNormalInitializer()); // rejtett réteg súly értéke
            y = CNTKLib.Sigmoid(CNTKLib.Times(w2, CNTKLib.Sigmoid(CNTKLib.Plus(CNTKLib.Times(w1, x), b))));
        }

        public void Train(string[] trainData)
        {
            int n = trainData.Length;

            // Extend graph // gráf kiegészítései
            Variable yt = Variable.InputVariable(new int[] { 1, outputSize }, DataType.Float);
            Function loss = CNTKLib.BinaryCrossEntropy(y, yt);

            Function y_rounded = CNTKLib.Round(y);
            Function y_yt_equal = CNTKLib.Equal(y_rounded, yt);

            Learner learner = CNTKLib.SGDLearner(new ParameterVector(y.Parameters().ToArray()), new TrainingParameterScheduleDouble(0.01, 1));
            Trainer trainer = Trainer.CreateTrainer(y, loss, y_yt_equal, new List<Learner>() { learner });

            // Train // tanulás
            for (int i = 1; i <= 50; i++)
            {
                double sumLoss = 0;
                double sumEval = 0;
                foreach (string line in trainData)
                {
                    float[] values = line.Split('\t').Select(x => float.Parse(x)).ToArray();
                    var inputDataMap = new Dictionary<Variable, Value>()
                    {
                        { x, LoadInput(values[0], values[1], values[2], values[3] )},
                        { yt, Value.CreateBatch(yt.Shape, new float[] { values[4] }, DeviceDescriptor.CPUDevice) }
                    };
                    var outputDataMap = new Dictionary<Variable, Value>() { { loss, null } };

                    trainer.TrainMinibatch(inputDataMap, false, DeviceDescriptor.CPUDevice);
                    sumLoss += trainer.PreviousMinibatchLossAverage();
                    sumEval += trainer.PreviousMinibatchEvaluationAverage();
                }
                Console.WriteLine(String.Format("{0}\tloss:{1}\teval:{2}", i, sumLoss / n, sumEval / n));
            }
        }

        public float Prediction(float szobak_szama, float terulet, float falu_varos, float ar) // előrejelzés az adatok alapján
        {
            var inputDataMap = new Dictionary<Variable, Value>() { { x, LoadInput(szobak_szama, terulet, falu_varos, ar) } };
            var outputDataMap = new Dictionary<Variable, Value>() { { y, null } };
            y.Evaluate(inputDataMap, outputDataMap, DeviceDescriptor.CPUDevice);
            return outputDataMap[y].GetDenseData<float>(y)[0][0];
        }

        Value LoadInput(float szobak_szama, float terulet, float falu_varos, float ar) // normalizálás
        {


            float[] x_store = new float[inputSize];
            x_store[0] = szobak_szama / 10;
            x_store[1] = terulet / 400;
            x_store[2] = falu_varos / 2;
            x_store[3] = ar / 400000;
            return Value.CreateBatch(x.Shape, x_store, DeviceDescriptor.CPUDevice);
        }
    }

    public class Program
    {
        string[] trainData = File.ReadAllLines(@"C:\Users\sznik\Desktop\szakertoi_rendszerek_beadandom\haz.txt"); // betölti az adatbázist
        NeuralNetwork app = new NeuralNetwork(); // létrehozza a neural network appot

        void Run() // meghívja a függvényeket
        {
            app.Train(trainData);
            FileTest();
            ConsoleTest();
        }

        void FileTest() // előrejelzés pontossága
        {
            int TP = 0, TN = 0, FP = 0, FN = 0;
            foreach (string line in trainData)
            {
                float[] values = line.Split('\t').Select(x => float.Parse(x)).ToArray();
                int good = (int)values[4];
                int pred = (int)Math.Round(app.Prediction(values[0], values[1], values[2], values[3]));

                if (pred == good)
                    if (pred == 1)
                        TP++;
                    else
                        TN++;
                else
                    if (pred == 1)
                    FP++;
                else
                    FN++;
            }
            float accuracy = (float)(TP + TN) / (TP + FP + TN + FN);
            float precision = (float)TP / (TP + FP);
            float sensitivity = (float)TP / (TP + FN);
            float F1 = 2 * (precision * sensitivity) / (precision + sensitivity);
            Console.WriteLine(String.Format("True positive:\t{0}\nTrue negative:\t{1}\nFalse positive:\t{2}\nFalse negative:\t{3}", TP, TN, FP, FN));
            Console.WriteLine(String.Format("Accuracy:\t{0}\nPrecision:\t{1}\nSensitivity:\t{2}\nF1 score:\t{3}", accuracy, precision, sensitivity, F1));
        }

        void ConsoleTest() // felhasználói bemenet
        {
            while (true)
            {
                Console.Write("Szobák száma:");
                float szobak_szama = float.Parse(Console.ReadLine());
                Console.Write("Terület (m2):");
                float terulet = float.Parse(Console.ReadLine());
                Console.Write("Falu(1) vagy Város(2):");
                float falu_varos = float.Parse(Console.ReadLine());
                Console.Write("A ház ára:");
                float ar = float.Parse(Console.ReadLine());

                //Console.WriteLine("Prediction:" + app.Prediction(szobak_szama, terulet, falu_varos, ar));
                Console.WriteLine("A házat " + app.Prediction(szobak_szama, terulet, falu_varos, ar) * 100 +"%-ban ajánlott megvenni.");

            }
        }

        static void Main(string[] args)
        {
            new Program().Run();
        }
    }
}