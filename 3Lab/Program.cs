using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic;

namespace CarIssuesInferenceApp
{
    class Program
    {
        static Random random = new Random();

        static void Main(string[] args)
        {
            Console.WriteLine("Sveiki atvykę į automobilių problemų sistemą!");
            Console.Write("Ar automobilis buvo reguliariai priežiūrėtas (taip/ne)? ");

            string maintenanceInput = Console.ReadLine()?.ToLower();
            bool regularMaintenance = maintenanceInput == "taip";

            InferenceEngine ie = new InferenceEngine();

            double flatTireBase = 0.15;
            double batteryFailureBase = 0.10;
            double engineOverheatBase = 0.05;
            double transmissionFailureBase = 0.03;
            double brakeFailureBase = 0.08;

            flatTireBase = GetRandomizedProbability(flatTireBase, regularMaintenance);
            batteryFailureBase = GetRandomizedProbability(batteryFailureBase, regularMaintenance);
            engineOverheatBase = GetRandomizedProbability(engineOverheatBase, regularMaintenance);
            transmissionFailureBase = GetRandomizedProbability(transmissionFailureBase, regularMaintenance);
            brakeFailureBase = GetRandomizedProbability(brakeFailureBase, regularMaintenance);

            Variable<bool> flatTire = Variable.Bernoulli(flatTireBase).Named("FlatTire");
            Variable<bool> batteryFailure = Variable.Bernoulli(batteryFailureBase).Named("BatteryFailure");
            Variable<bool> engineOverheat = Variable.Bernoulli(engineOverheatBase).Named("EngineOverheat");
            Variable<bool> transmissionFailure = Variable.Bernoulli(transmissionFailureBase).Named("TransmissionFailure");
            Variable<bool> brakeFailure = Variable.Bernoulli(brakeFailureBase).Named("BrakeFailure");

            Variable<bool> regularMaintenanceVar = Variable.Bernoulli(regularMaintenance ? 0.7 : 0.2).Named("RegularMaintenance");

            Variable<bool> flatTireWithMaintenance = Variable.New<bool>().Named("FlatTireWithMaintenance");
            Variable<bool> batteryFailureWithMaintenance = Variable.New<bool>().Named("BatteryFailureWithMaintenance");
            Variable<bool> engineOverheatWithMaintenance = Variable.New<bool>().Named("EngineOverheatWithMaintenance");
            Variable<bool> transmissionFailureWithMaintenance = Variable.New<bool>().Named("TransmissionFailureWithMaintenance");
            Variable<bool> brakeFailureWithMaintenance = Variable.New<bool>().Named("BrakeFailureWithMaintenance");

            using (Variable.If(regularMaintenanceVar))
            {
                flatTireWithMaintenance.SetTo(Variable.Bernoulli(GetRandomizedProbability(0.10, true)));
                batteryFailureWithMaintenance.SetTo(Variable.Bernoulli(GetRandomizedProbability(0.07, true)));
                engineOverheatWithMaintenance.SetTo(Variable.Bernoulli(GetRandomizedProbability(0.03, true)));
                transmissionFailureWithMaintenance.SetTo(Variable.Bernoulli(GetRandomizedProbability(0.02, true)));
                brakeFailureWithMaintenance.SetTo(Variable.Bernoulli(GetRandomizedProbability(0.05, true)));
            }

            using (Variable.IfNot(regularMaintenanceVar))
            {
                flatTireWithMaintenance.SetTo(flatTire);
                batteryFailureWithMaintenance.SetTo(batteryFailure);
                engineOverheatWithMaintenance.SetTo(engineOverheat);
                transmissionFailureWithMaintenance.SetTo(transmissionFailure);
                brakeFailureWithMaintenance.SetTo(brakeFailure);
            }

            using (Variable.If(batteryFailureWithMaintenance))
            {
                engineOverheatWithMaintenance.SetTo(Variable.Bernoulli(GetRandomizedProbability(0.15, regularMaintenance)));
            }

            using (Variable.IfNot(batteryFailureWithMaintenance))
            {
                engineOverheatWithMaintenance.SetTo(engineOverheat);
            }

            Variable<bool> accident = Variable.New<bool>().Named("Accident");

            accident.SetTo(brakeFailureWithMaintenance | engineOverheatWithMaintenance | transmissionFailureWithMaintenance | batteryFailureWithMaintenance | flatTireWithMaintenance);

            Dictionary<string, object> results = new Dictionary<string, object>
            {
                { "Kiaura padanga", ie.Infer(flatTireWithMaintenance) },
                { "Akumuliatoriaus gedimas", ie.Infer(batteryFailureWithMaintenance) },
                { "Variklio perkaitimas", ie.Infer(engineOverheatWithMaintenance) },
                { "Pavarų dėžės gedimas", ie.Infer(transmissionFailureWithMaintenance) },
                { "Stabdžių gedimas", ie.Infer(brakeFailureWithMaintenance) },
                { "Avarijos rizika", ie.Infer(accident) }
            };

            string resultsText = "\n** Išvada **\n";
            foreach (var result in results)
            {
                string prob = result.Value switch
                {
                    Bernoulli b => $"{b.GetMean() * 100:F2}%",
                    Gaussian g => $"{g.GetMean() * 100:F2}%",
                    _ => "Nežinoma"
                };
                resultsText += $"{result.Key}: {prob}\n";
            }

            Console.WriteLine(resultsText);
            Console.WriteLine("\n** Informacija **");

            if (results["Avarijos rizika"] is Bernoulli accidentResult && accidentResult.GetMean() > 0.5)
            {
                Console.WriteLine("Įspėjimas! Yra labai didelė avarijos rizika. Prašome vairuoti atsargiai!");
            }
            else
            {
                Console.WriteLine("Avarijos rizika yra maža, tačiau būtina būti atsargiems ir tinkamai prižiūrėti savo automobilį.");
            }
        }

        static double GetRandomizedProbability(double baseProbability, bool isRegularMaintenance)
        {
            double variation = random.NextDouble() * 0.1 - 0.05;
            double randomizedProbability = baseProbability + variation;

            if (!isRegularMaintenance)
            {
                randomizedProbability += 0.1;
            }

            return Math.Max(0, Math.Min(1, randomizedProbability));
        }
    }
}