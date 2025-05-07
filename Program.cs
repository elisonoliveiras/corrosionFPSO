using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Globalization;
using Python.Runtime;
using System.Collections.Generic;

class Program
{
    static void Main(string[] args)
    {
        if (args.Length == 0)
        {
            Console.WriteLine("Por favor, especifique o modo de operacao: 'train' ou 'predict'");
            return;
        }

        var mode = args[0].ToLower();
        if (mode != "train" && mode != "predict")
        {
            Console.WriteLine("Modo invalido. Use 'train' ou 'predict'");
            return;
        }

        Console.WriteLine($"Iniciando programa no modo: {mode}");

        try
        {
            ConfigurarAmbiente();
            using (Py.GIL())
            {
                if (mode == "train")
                {
                    TrainModel();
                }
                else
                {
                    Predict();
                }
            }
        }
        catch (Exception ex)
        {
            File.WriteAllText("error.log", $"Erro: {ex.Message}\nStack trace: {ex.StackTrace}");
            throw;
        }
        finally
        {
            try
            {
                PythonEngine.Shutdown();
            }
            catch (NotSupportedException)
            {
                // Ignorar exceção de BinaryFormatter desativado
            }
        }
    }

    static void ConfigurarAmbiente()
    {
        var culture = new CultureInfo("en-US");
        CultureInfo.DefaultThreadCurrentCulture = culture;
        CultureInfo.DefaultThreadCurrentUICulture = culture;

        // Autodetect Python DLL (python39.dll) via PATH
        var paths = Environment.GetEnvironmentVariable("PATH").Split(Path.PathSeparator);
        string pythonDll = null;
        foreach (var p in paths)
        {
            var dllPath = Path.Combine(p, "python39.dll");
            if (File.Exists(dllPath))
            {
                pythonDll = dllPath;
                break;
            }
        }
        if (pythonDll != null)
        {
            Runtime.PythonDLL = pythonDll;
        }
        else
        {
            Console.WriteLine("Aviso: python39.dll não encontrado no PATH. Certifique-se de que o Python esteja instalado e no PATH.");
        }

        PythonEngine.Initialize();
    }

    static void TrainModel()
    {
        using (Py.GIL())
        {
            try
            {
                Console.WriteLine("Executando script de treinamento...");
                
                dynamic sys = Py.Import("sys");
                dynamic os = Py.Import("os");
                
                // Configurar o ambiente Python
                sys.path.append(os.getcwd());
                
                // Importar e executar o script diretamente
                dynamic train_module = Py.Import("train_model");
                
                Console.WriteLine("Treinamento concluído com sucesso!");
            }
            catch (PythonException px)
            {
                Console.WriteLine($"Erro Python: {px.Message}");
                Console.WriteLine($"Stack trace Python: {px.StackTrace}");
                File.WriteAllText("error.log", $"Erro Python: {px.Message}\nStack trace: {px.StackTrace}");
                throw;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Erro durante o treinamento: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
                File.WriteAllText("error.log", $"Erro durante o treinamento: {ex.Message}\nStack trace: {ex.StackTrace}");
                throw;
            }
        }
    }

    static void Predict()
    {
        using (Py.GIL())
        {
            try
            {
                dynamic np = Py.Import("numpy");
                dynamic pd = Py.Import("pandas");
                dynamic joblib = Py.Import("joblib");
                dynamic tf = Py.Import("tensorflow");

                Console.WriteLine("Carregando modelo e preprocessadores...");
                
                dynamic model = tf.saved_model.load("saved_model");
                Console.WriteLine("Modelo carregado com sucesso");

                dynamic scaler = joblib.load("scaler.pkl");
                dynamic encoders = joblib.load("encoders.pkl");
                Console.WriteLine("Scaler e encoders carregados com sucesso");

                Console.WriteLine("Carregando dados para previsao...");
                var args = new PyTuple(new PyObject[] { new PyString("dados_para_prever.csv") });
                var kwargs = new PyDict();
                kwargs["sep"] = new PyString(";");
                kwargs["encoding"] = new PyString("latin1");
                kwargs["decimal"] = new PyString(",");
                dynamic df = pd.InvokeMethod("read_csv", args, kwargs);
                Console.WriteLine($"Dados carregados com sucesso. Shape: {df.shape}");

                Console.WriteLine("Colunas do DataFrame:");
                int idx = 1;
                foreach (var col in df.columns)
                {
                    Console.WriteLine($"Feature {idx} -> {col}");
                    idx++;
                }

                var numericColumns = new[] { "posição_longitudinal", "altura", "anos_vida", "anos_coat", "flag" };
                var categoricalColumns = new[] { "seção_carga_interface", "elemento" };

                try
                {
                    foreach (var col in categoricalColumns)
                    {
                        Console.WriteLine($"Processando coluna: {col}");
                        try
                        {
                            df[col] = encoders[col].transform(df[col]);
                        }
                        catch (PythonException px)
                        {
                            Console.WriteLine($"Aviso: categoria não vista na coluna {col}: {px.Message}");
                            throw;
                        }
                        Console.WriteLine($"Valores únicos em {col} após encoding: {string.Join(", ", df[col].unique().tolist())}");
                        Console.WriteLine($"Min de {col}: {df[col].min()}, Max de {col}: {df[col].max()}, Média de {col}: {df[col].mean()}");
                    }

                    if (df["flag"].dtype == "object")
                    {
                        Console.WriteLine("Convertendo flag para binario...");
                        dynamic flagMap = new PyDict();
                        flagMap.SetItem("Sim", new PyInt(1));
                        flagMap.SetItem("Nao", new PyInt(0));
                        df["flag"] = df["flag"].map(flagMap);
                        var flag = df["flag"];
                        Console.WriteLine($"Valores unicos apos conversao: {flag.unique().tolist()}");
                    }

                    var allColumns = numericColumns.Concat(categoricalColumns).ToArray();
                    dynamic pyColumns = new PyList();
                    foreach (var col in allColumns)
                    {
                        pyColumns.Append(new PyString(col));
                    }
                    Console.WriteLine("Extraindo features...");
                    dynamic features = df[pyColumns].to_numpy();
                    Console.WriteLine($"Shape das features: {features.shape}");
                    
                    Console.WriteLine("Normalizando features...");
                    // Separar features numéricas e categóricas
                    dynamic numeric_indices = np.array(new[] { 0, 3, 4, 5 });
                    dynamic cat_indices = np.array(new[] { 1, 2 });
                    dynamic bin_indices = np.array(new[] { 6 });
                    
                    dynamic features_numeric = features.take(numeric_indices, 1);
                    dynamic features_cat = features.take(cat_indices, 1);
                    dynamic features_bin = features.take(bin_indices, 1);
                    
                    // Normalizar apenas as features numéricas
                    dynamic features_scaled_numeric = scaler.transform(features_numeric);
                    
                    // Criar lista de arrays para concatenar
                    dynamic arrays_to_concat = new PyList();
                    arrays_to_concat.append(features_scaled_numeric);
                    arrays_to_concat.append(features_cat);
                    arrays_to_concat.append(features_bin);
                    
                    // Combinar todas as features
                    features = np.hstack(arrays_to_concat);
                    Console.WriteLine($"Shape após normalização: {features.shape}");

                    Console.WriteLine("Fazendo previsoes...");
                    dynamic serving_default = model.signatures["serving_default"];
                    
                    // Criar lista para armazenar as previsões
                    dynamic predictions_list = new PyList();
                    
                    // Fazer previsões em lotes de 32 amostras
                    int batch_size = 32;
                    int total_samples = (int)features.shape[0];
                    
                    for (int i = 0; i < total_samples; i += batch_size)
                    {
                        int end_idx = Math.Min(i + batch_size, total_samples);
                        dynamic batch = features[np.arange(i, end_idx)];
                        dynamic tensor = tf.convert_to_tensor(batch);
                        tensor = tf.cast(tensor, tf.float32);
                        dynamic result = serving_default(tensor);
                        dynamic batch_predictions = result["output_0"].numpy();
                        predictions_list.append(batch_predictions);
                    }
                    
                    // Concatenar todas as previsões
                    dynamic predictions = np.concatenate(predictions_list);
                    Console.WriteLine($"Shape das previsoes: {predictions.shape}");

                    Console.WriteLine("Salvando previsoes...");
                    dynamic pyColumns2 = new PyList();
                    pyColumns2.Append(new PyString("C1"));
                    pyColumns2.Append(new PyString("C2"));
                    
                    args = new PyTuple(new PyObject[] { predictions });
                    kwargs = new PyDict();
                    kwargs["columns"] = pyColumns2;
                    dynamic predictionsDf = pd.InvokeMethod("DataFrame", args, kwargs);
                    
                    // Juntar as previsões ao DataFrame original de entrada
                    df["C1"] = predictionsDf["C1"];
                    df["C2"] = predictionsDf["C2"];

                    // Antes de salvar o DataFrame completo com as previsões, fazer o desencoding das colunas categóricas
                    // Reverter seção_carga_interface
                    if (((dynamic)encoders).__contains__("seção_carga_interface"))
                    {
                        var le = encoders["seção_carga_interface"];
                        df["seção_carga_interface"] = le.inverse_transform(df["seção_carga_interface"].astype("int"));
                    }
                    // Reverter elemento
                    if (((dynamic)encoders).__contains__("elemento"))
                    {
                        var le = encoders["elemento"];
                        df["elemento"] = le.inverse_transform(df["elemento"].astype("int"));
                    }
                    // Salvar o DataFrame completo com as previsões no formato original
                    var argsCompleto = new PyTuple(new PyObject[] { new PyString("previsoes_completas.csv") });
                    var kwargsCompleto = new PyDict();
                    kwargsCompleto["sep"] = new PyString(";");
                    kwargsCompleto["index"] = false.ToPython();
                    kwargsCompleto["decimal"] = new PyString(",");
                    df.InvokeMethod("to_csv", argsCompleto, kwargsCompleto);

                    Console.WriteLine("Previsoes concluidas com sucesso!");
                }
                catch (Exception ex)
                {
                    File.WriteAllText("error.log", $"Erro durante o processamento: {ex.Message}\nStack trace: {ex.StackTrace}");
                    throw;
                }
            }
            catch (PythonException px)
            {
                File.WriteAllText("error.log", $"Erro Python: {px.Message}");
                throw;
            }
            catch (Exception ex)
            {
                File.WriteAllText("error.log", $"Erro durante a previsao: {ex.Message}");
                throw;
            }
        }
    }
} 