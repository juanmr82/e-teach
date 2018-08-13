'''
Created on Aug 10, 2018
@author: Juan Reina
@contact: juanmr82@gmail.com

This module uses tensorflow on a dataset to implement a multivarian linear regression.
The following input arguments are needed and for practical purposes, in CSV format
and only float values
    1. File name. Must be specified with -i
    2. Column number to be used as Output. It must specified with -y 
    3. Learning rate. It must be specified with -a
    4. Nr of training epochs, specified with -t
    5. If file contains header with -H
    
In this version, the script doesnt do model validation or data plotting,
it is simply a demonstration of Tensorflow to quickly iterate through a 
CSV file, the use of the Data APU and iterators

'''

from com.pybsoft.eteach.regression import *
import getopt
import sys
import pandas as pd


def get_arguments(argv):
    '''
    @return:  Key Map containing the user arguments 
    '''
    argumentMap = {}
    try:
        ops, args = getopt.getopt(argv, "hHei:y:a:t:", [])
    except getopt.GetoptError:
        print("Error in arguments")
        print("mvlinear.py -i <PAHT to INPUT FILE> -y <Answer/Output Column number> -a <learning rate>")
        sys.exit(2)
    if (ops.__len__() == 0):
        print("No arguments were specified. Please use the sctipt like this:")
        print("mvlinear.py -i <PAHT to INPUT FILE> -y <Answer/Output Column number> -a <learning rate>")
        sys.exit(2)
    for op, arg in ops:
        if (op == "-h"):
            print("\n"
                  "Usage:\n"
                  "mvlinear.py -i <PAHT to INPUT FILE> -y <Answer/Output Column number> -a <learning rate>\n"
                  "-i File Name with input data.\n"
                  "-y Within the file, which column has the output/result for the regression\n"
                  "-a Learning rate. Must be a real value\n"
                  "\n")
            sys.exit()
        if (op == "-h"):
            argumentMap["header"] = True
        if (op == "-i"):
            argumentMap["file_path"] = arg
        if (op == "-t"):
            try:
                argumentMap["epochs"] = int(arg)
                if(argumentMap["epochs"]<1):
                    print("Error. Nr of epochs must be a natural number greater than zero")
                    print("Finishing script")
                    sys.exit(2)
            except ValueError:
                print("Error. Nr of epochs must be a natural number greater than zero")
                print("Finishing script")
                sys.exit(2)
        if (op == "-y"):
            try:
                y = int(arg)
                if (y <= 0):
                    print("Output column index/number cant be less than zero")
                    sys.exit(2)
                argumentMap["y_col_nr"] = y
            except ValueError:
                print("Error. Output column index/number must be integer!!!!")
                print("Finishing script")
                sys.exit(2)
        if (op == "-a"):
            try:
                a = float(arg)
                argumentMap["alpha"] = a
            except ValueError:
                print("Error. Output column index/number must be integer/float!!!!")
                print("Finishing script")
                sys.exit(2)
    # Check if the arguments are correct
    if (argumentMap.get("file_path", None) is None):
        print("Error. File Path was not specified")
        print("Finishing script")
    if (argumentMap.get("y_col_nr", None) is None):
        print("Error. Output/Answer Column number was not specified")
        print("Finishing script")
    if (argumentMap.get("alpha", None) is None):
        print("Error. Learning rate was not specified")
        print("Finishing script")
    if (argumentMap.get("header", None) is None):
        argumentMap["header"]=False
    return argumentMap


    
    
def pack_features(features, labels):
    '''
    This code is based on the function with the same name
    in the Custom training: walkthrough section in the 
    Tensorflow website
    @return: Pack of features and label as a stacked tensor instead as a dictionary of tensors
    '''
    #My addition to the function. Cast data to float32
    for k,v in features.items():
        features[k] = tf.cast(features[k], dtype=tf.float32, name=k) 
             
    features = tf.stack(list(features.values()), axis=1)
    return features, labels



def main(argv):
    '''
    Defining a main function, in my opinion, can help other people to understand my code
    '''
    # Get the arguments from the console and return a key-mmap
    arguments = get_arguments(argv)
    
    # Ath this point the arguments syntaxt is correct. The script doesnt know yet if the
    # file does actually exists, has the correct number of columns and the correct format
    # Following assumptions are done:
    # 1. The first row contains the column names
    # 2. It is CSV Format
    # 3. All data are integer/float
    print("STARTING SCRIPT")
    print("")
    
    print("Analyzing the input data file at ",arguments["file_path"])
    
    data_file=None
    nr_rows,nr_columns = (0,0)
    mean_values,max_values,min_values=(None,None,None)
    
    try:
        data_file = pd.read_csv(arguments["file_path"],sep='\s+|\t+|,|;',engine='python',header=None)
        #Get the number of files and columns, and the min, max and mean of each separated column
        nr_rows,nr_columns =data_file.shape
        mean_values = data_file.iloc[:,:].mean()
        max_values = data_file.iloc[:,:].max()
        min_values = data_file.iloc[:,:].min()
        
    except:
        print("Error reading file ",arguments["file_path"], "for processing")
        print("FINISHING SCRIPT")
        sys.exit(2)

    #Displaying the data in a user friendly way
    print("Summary of data:" )
    print("File header: ",arguments["header"])
    print("Nr Rows: %d and Nr Columns: %d" % (nr_rows,nr_columns))
    print("Mean value per column:",[mean_values[i] for i in range(nr_columns)])
    print("Max value per column:", [max_values[i] for i in range(nr_columns)])
    print("Min value per column:",[min_values[i] for i in range(nr_columns)])
    print("")
        
    
    print("Defining the data import strategy, the model and its optimizer")
    
    #Inner FUnctions for the normalization of the data
    def normalize_data(features, labels):
        '''
        Mapping function to define feature normalization 
        '''
        i = 0 
        for k,v in features.items():
            features[k] = (features[k]-mean_values[i])/(max_values[i]-min_values[i])
            i = i+1     
        labels = (labels-mean_values[nr_columns-1])/(max_values[nr_columns-1]-min_values[nr_columns-1])
        return features, labels
        '''
        End of Function
        '''  
    
    '''
    @todo:  Improve this very lazy Dataset batch size selection strategy or let the user to choose its own batch size    
    '''
    if(nr_rows>1000):
        batch= 320
    elif(nr_rows >100):
        batch = 32
    elif(nr_rows>1):
        batch = 1
    
    #Preparing Dataset Carachteristics
    label_name = "Y"
    col_names = ["Col%d"%(i) for i in range(nr_columns)]
    col_names[nr_columns-1] = label_name  
    features_names = col_names[:-1]
    
    '''
    @todo: Improve the delimiter selection. Probably a regex
    '''
    #Creating Dataset from CSV File and adding pre-processing info
    dataset = tf.contrib.data.make_csv_dataset(arguments["file_path"],
                                               batch_size=batch, 
                                               shuffle=False,
                                               num_epochs=1,
                                               column_names=col_names,
                                               label_name=label_name,
                                               header=arguments["header"],
                                               field_delim='\t')
    #dataset = dataset.batch(batch,drop_remainder=True)
    dataset = dataset.map(normalize_data)
    dataset = dataset.map(pack_features)
    
    '''
    In this part of the code, the Model, the iterator through the file and the optimizer are defined
    '''
    
    #Creating iterator though the data
    #Initializable allow us to re-initialize this iterator after each epoch
    iterator = dataset.make_initializable_iterator()
    X,Y = iterator.get_next()
    
    W = tf.Variable([[nmp.random.rand() for i in range(nr_columns-1)]],dtype=tf.float32,name="WeightMatrix")
    b = tf.Variable(tf.zeros([1]),dtype=tf.float32,name="bias")

    #The model 
    hypothesis = tf.matmul(W, X, transpose_b=True) + b
    
    #Necessary to allow the difference of matrixes when batch > 1
    Y = tf.transpose(Y)
    
    #Cost/loss function
    cost_function = tf.reduce_sum(tf.squared_difference(Y,hypothesis))/(2*nr_rows)
    
    #Optimizer function with learning rate
    grad_descent = tf.train.GradientDescentOptimizer(arguments["alpha"]).minimize(cost_function)
    
    #Initializer
    init_vars = tf.global_variables_initializer()

    '''
    The definition of the Model and nodes is over. Now comes the session definition
    '''
    
    print("Starting training session")
    with tf.Session() as sess:
    
        
        print("Initializationg variables")
        sess.run(init_vars)
        print("Initial values of linear model:")
        print("W:",sess.run(W))
        print("b:",sess.run(b))
        
        print("Running the model with ",arguments["epochs"], "epochs")
        i = 0
        j=0
        cost = 0

        '''
        Although the number of epochs steps can be defined in the dataset definition, 
        I left it there as 1:
        The reason for this is that this gives me the possibility to signal the end of 
        the iteration of the file with a tf.errors.OutOfRangeError exception and store/show
        the cost/loss value for plotting or to store it on a CSV output file
        '''
        for i in range(arguments["epochs"]):
            #Start/Re-start iterator
            sess.run(iterator.initializer)
            
            while True:
                try:
                    _,cost,ys,yt=sess.run([grad_descent,cost_function,Y,hypothesis])

                except tf.errors.OutOfRangeError:
                    #suma = suma/(2*nr_rows)
                    if(i%50 ==0):
                        print("Epoch ",i,"ended with loss/cost value of ",cost )
                       
                    break
                
                
            i=i+1
                
        print("")
        print("Fininshing with cost/loss value of", cost)
        print("Final values of linear model:")
        print("W:",sess.run(W))
        print("b:",sess.run(b))
        
        sess.close()
       
    print("FINISHING SCRIPT")
    
    '''
    @todo: Model validation
    @todo: Plotting of cost function values
    @todo: Exporting training session statistics into a CSV file or serialize it as a JSON Object 
    '''
    
# EXECUTION STARTS HERE
if __name__ == "__main__":
    main(sys.argv[1:])
