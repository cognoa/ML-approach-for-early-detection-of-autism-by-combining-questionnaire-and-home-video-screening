import itertools
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from sklearn import preprocessing
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from sklearn import cross_validation
from sklearn import feature_extraction
import math
import random
import copy as cp
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn_0p18_bridge_code import *


#converts a value to a float gracefully
def convert_to_float(x):
    try:
        return float(x)
    except:
        return np.NaN

   
#handy function to return list of DataFrame columns that have a keyword (like the insturment name) somewhere in their title
def columns_about(df, keyword):
    return [x for x in list(df.columns) if keyword.lower() in x.lower()]
    

#handy function to replace certain values in certain columns of a dataframe. Useful for feature value mapping before training
def replace_values_in_dataframe_columns(df, columns, values, replacement, replace_if_equal=True):

    for column in columns:
        
        if (replace_if_equal):
            mask = df[column].isin(values)
        else:
            mask = np.logical_not( df[column].isin(values) )
            
        df[column][mask] = replacement
   
#handy function to subsample dataframe by choosing x% of the samples of each 'class' as defined by a column in the df                
def subsample_per_class(df, class_column_name, dict_ratio_per_class):
    output_df = pd.DataFrame()
    for class_name in dict_ratio_per_class.keys():
        df_this_class = df[df[class_column_name]==class_name]
        ratio = dict_ratio_per_class[class_name]
        total = len(df_this_class)
        sample_size = int(float(total) * ratio)
        subset = df_this_class.loc[np.random.choice(df_this_class.index, sample_size, replace=False)]
        subset = subset.reset_index()
        output_df = pd.concat([output_df, subset])
    return output_df.reset_index(drop=True)


def balance_dataset_on_dimensions(dataset, dimensions, enforce_group_weight_instructions=None, verbose=False):

    ''' if only dataset and dimensions are passed, this function will enforce equal weighting for the sum of all
    rows in all combinations of dimensions.
	....... Note: many new functions are added lower in this file for purposes of constructing the enforce_group_weight_instructions
	....... or the inputs that are needed to determine it, 
	....... such as get_desired_condition_fracs, get_general_diagnoses_from_specific_diagnoses, get_combined_column_from_one_hot_columns

    If enforce_group_weight_instructions is defined, then this function will first use dimensions to make a set of equal weights,
    then make a new set of weights using enforce_group_weight_instructions, and will multiply the two sets of weights together

    The specification of the format of enforce_group_weight_instructions is:
        enforce_group_weight_instructions = {
            'balance_across_this_key': balance_column_name,
            'bin_by': bin_by_column_name,
            'desired_tot_weights_df: desired_tot_weights_df
        }
        ... Where balance_column_name is a column of dataset that tells you what column needs to be re-balanced (example: 'condition')
        ... Where bin_by_column_name tells you the bins across which this re-balancing should be done (example: 'age_category'). Need not match
            any of the dimensions of the initial balancing.
        ... Where desired_fracs_df is a dataframe that provides the desired output fractional weights, with rows representing the bins
            and columns representing the instructions for each possible value of the balance_column_name in that bin.

    Here is an example of enforce_group_weight_instructions:
        enforce_group_weight_instructions = {
            'balance_across_this_key': 'condition',
            'bin_by': 'age_category',
            'desired_tot_weights_df': desired_tot_weights_df
        }
        desired_tot_weights_df = pd.DataFrame({
            'age_category': ['3-', '4+'],
            'autism': [0.5, 0.5],
            'ADHD': [0.1, 0.3],
            'neurotypical': [0.4, 0.2]
        })

    '''

    if 'sample_weights' in dataset.columns:
        dataset = dataset.drop('sample_weights', 1)
    if 'pre_scaled_sample_weights' in dataset.columns:
        dataset = dataset.drop('pre_scaled_sample_weights', 1)
    counts = {}
    weights = {}
    total_count = 0
    for  index, row in dataset.iterrows():
        key = tuple([row[dimension] for dimension in dimensions])
        counts[key] = counts[key]+1 if key in counts else 1
        total_count += 1

    for key,count in counts.iteritems():
        weight = float(total_count) / float(count)
        weights[key] = weight
        
    if verbose:
        for group in counts:
            print(str(group)+": "+str(counts[group])+" out of "+str(total_count)+" -> weight "+str(weights[group]))

    sample_weight_dict = {}
    for  index, row in dataset.iterrows():

        sample_weight =  weights[tuple([row[dimension] for dimension in dimensions])]
        sample_weight_dict[index] = sample_weight
    dataset['sample_weights'] = pd.Series(sample_weight_dict)

    #### The rest of this function optionally enforces a specific requested normalization of sample weights,
    #### According to enforce_group_weight_instructions
    def weight_scaling_function(row, enforce_group_weight_instructions, weight_scaling_factors_by_bin):
        ''' helper function for balance_dataset_on_dimensions, which determines scaling factors needed 
        to achieve enforce_group_weight_instructions '''
        balance_across_this_key = enforce_group_weight_instructions['balance_across_this_key']
        bin_by = enforce_group_weight_instructions['bin_by']
        current_weight = row['sample_weights']
        if row[bin_by] not in weight_scaling_factors_by_bin.keys():
            raise ValueError('Error, weights for grouping '+row[bin_by]+' not understood')
        scaling_factors = weight_scaling_factors_by_bin[row[bin_by]]
        this_group = row[balance_across_this_key]
        scaling_factor = scaling_factors[this_group]
        scaled_weight = current_weight*scaling_factor
        return scaled_weight

    if enforce_group_weight_instructions is not None:
        balance_across_this_key = enforce_group_weight_instructions['balance_across_this_key']
        allowed_groups_to_balance = np.unique(dataset[balance_across_this_key].values)
        ### balance_across_this_key might be a column that specifies non-overlapping conditions,
        ### or any other grouping column else you want to fix weights on
        bin_by = enforce_group_weight_instructions['bin_by']
        ### An example of bin_by would be age_category. Will enforce weighting according to
        ### instructions in each bin.
        unique_bins = np.unique(dataset[bin_by].values)
        dataset['pre_scaled_sample_weights'] = cp.deepcopy(dataset['sample_weights'].values)
        weight_scaling_factors_by_bin = {}
        ### Now loop over bins and determine needed scaling factors
        for this_bin in unique_bins:
            this_bin_dataset = dataset[dataset[bin_by]==this_bin]
            desired_fracs_df = enforce_group_weight_instructions['desired_tot_weights_df']
            ### desired_bin_fracs are the weights we want to enforce in this bin
            desired_bin_fracs = (desired_fracs_df[desired_fracs_df[bin_by]==this_bin][allowed_groups_to_balance].reset_index()).iloc[0]
            starting_weights = (this_bin_dataset.groupby(balance_across_this_key).sum())['pre_scaled_sample_weights']
            sum_starting_weights = np.sum(starting_weights.values)
            starting_fractions = starting_weights / sum_starting_weights
            scaling_factors = desired_bin_fracs / starting_fractions
            weight_scaling_factors_by_bin[this_bin] = scaling_factors

        ### Now that we know the scaling factors, apply them
        dataset['sample_weights'] = dataset.apply(
                    weight_scaling_function, args=(enforce_group_weight_instructions, weight_scaling_factors_by_bin), axis=1)
    ### Extract the weights we want to return into a separate series and drop the redundant columns in the dataframe
    weights_to_return = cp.deepcopy(dataset['sample_weights'])
    if 'sample_weights' in dataset.columns:
        dataset = dataset.drop('sample_weights', 1)
    if 'pre_scaled_sample_weights' in dataset.columns:
        dataset = dataset.drop('pre_scaled_sample_weights', 1)

    return weights_to_return




#prepare a  dataset for modeling by preprocessing every feature into the appropriate encoding and splitting into two matrices
#X=features and Y=target
def prepare_data_for_modeling(df, feature_columns, feature_encoding_map, target_column, force_encoded_features=[]):
    ''' force_features is an optional argument when you want to force agreement with particular encoded feature set '''
    ######## 
    #### motivation for "pseudo mixed ordinal/categorical" features:
    #### values 0 - 4 are considered to be ordered from low to high severity
    #### however other values and missing values could have all kinds of different meanings
    #### in rare cases it is obvious that a value like 8 can be considered "more sever" than
    #### a lower value, but this it is more common that no clear interpretation like this can be made
    ####
    #### Goal: make it easy for decision trees to track the severity of the values (0-4) by making feature numeric
    #### however, other "categorical" values exist. These should be clustered together on one side of the real 
    #### numerical values to make it as easy as possible for a decision tree to branch between them and the numerical values.
    #### The most common "categorical" values are those like 7 and 8. Map negative values (values exist as -1, -5, and -8) to a positive value
    #### by adding 20 to the value
    #### non numeric values such as '' are mapped to +50
    #### 
    #### These choices are pretty arbitrary, but they keep the categorical variables on one side of the distribution without merging any of them
    ########

    def safe_convert_to_number(x, dtype=float, problem_val=999):
        try:
            if pd.isnull(x): return problem_val
            converted_val = dtype(x)
            return converted_val
        except:
            return problem_val
            


    def num_cat_XForm(inValue, minVal=-0.0001):
        try:
            
            outValue = float(inValue) if float(inValue) > -0.0001 else float(inValue)+20
        except:
            ### Values like '' cannot be converted to floats and will end up here
            outValue = 50.
        return outValue

    mixed_numeric_categorical_features = [x for x in feature_columns if (x in feature_encoding_map and feature_encoding_map[x]=='mixed_numeric_categorical')]
    mixed_numeric_categorical_X = df[mixed_numeric_categorical_features]
    for column in mixed_numeric_categorical_X.columns:
        mixed_numeric_categorical_X[column].apply(num_cat_XForm)


    #scalar features don't require any enconding
    scalar_encoded_features = [x for x in feature_columns if (x in feature_encoding_map and feature_encoding_map[x]=='scalar')]
    scalar_encoded_X = df[scalar_encoded_features]
    for column in scalar_encoded_X.columns:
        scalar_encoded_X[column] = scalar_encoded_X[column].apply(safe_convert_to_number)

    #one_hot_encoding features are handled using DictVectorizer
    one_hot_encoded_features = [x for x in feature_columns if (x in feature_encoding_map and feature_encoding_map[x]=='one_hot')]
    one_hot_encoded_feature_dataset = df[one_hot_encoded_features]
    vectorizer = feature_extraction.DictVectorizer(sparse = False)
    one_hot_encoded_feature_dataset_dict = one_hot_encoded_feature_dataset.T.to_dict().values()  ### this becomes a list of dicts of rows
    one_hot_encoded_feature_dataset_dict_vectorized = vectorizer.fit_transform( one_hot_encoded_feature_dataset_dict )   ### This becomes a 2D array
    one_hot_encoded_feature_dataset = pd.DataFrame(one_hot_encoded_feature_dataset_dict_vectorized, columns=vectorizer.feature_names_)  
    one_hot_encoded_features = vectorizer.feature_names_
    #exclude all features that include the word "missing"
    one_hot_encoded_features = [x for x in one_hot_encoded_features if "missing" not in x]    
    #prepare X    
    one_hot_encoded_X=one_hot_encoded_feature_dataset[one_hot_encoded_features]
     
    #discrete encoding features is handled manually
    discrete_encoded_features = [x for x in feature_columns if (x in feature_encoding_map and feature_encoding_map[x]=='discrete')]
    discrete_encoded_X = df[discrete_encoded_features]
    for feature in discrete_encoded_X.columns:
        possible_values = discrete_encoded_X[feature].unique()
        discrete_encoded_X[feature+"=0"] = discrete_encoded_X[feature].apply(lambda x: 1 if x=='0' else 0)
        discrete_encoded_X[feature+"=1+"] = discrete_encoded_X[feature].apply(lambda x: 1 if (x=='1' or x=='2' or x=='3' or x=='4') else 0)
        if ('2' in possible_values) or ('3' in possible_values) or ('4' in possible_values):
            discrete_encoded_X[feature+"=2+"] = discrete_encoded_X[feature].apply(lambda x: 1 if (x=='2' or x=='3' or x=='4') else 0)
        if ('3' in possible_values) or ('4' in possible_values):
            discrete_encoded_X[feature+"=3+"] = discrete_encoded_X[feature].apply(lambda x: 1 if (x=='3' or x=='4') else 0)
        if '4' in possible_values:
            discrete_encoded_X[feature+"=4"] = discrete_encoded_X[feature].apply(lambda x: 1 if (x=='4') else 0)
    	discrete_encoded_X = discrete_encoded_X.drop(feature, axis=1)
    discrete_encoded_features = [x for x in discrete_encoded_X.columns]

    #presence of behavior features
    presence_of_behavior_encoded_features = [x for x in feature_columns if (x in feature_encoding_map and feature_encoding_map[x]=='presence_of_behavior')]
    presence_of_behavior_encoded_X = df[presence_of_behavior_encoded_features]
    presence_of_behavior_rules_dict = get_presence_of_behavior_rules_dict()
    for feature in presence_of_behavior_encoded_X.columns:
        if feature in presence_of_behavior_rules_dict.keys():
            presence_of_behavior_encoded_X[feature+"_behavior_present"] = presence_of_behavior_encoded_X[feature].apply(lambda x: 1 if x in presence_of_behavior_rules_dict[feature] else 0)
        else:
            print 'Warning: presence encoding rules not defined for this feature, so set to zero. Feature=', feature
            presence_of_behavior_encoded_X[feature+"_behavior_present"] = 0
    	presence_of_behavior_encoded_X = presence_of_behavior_encoded_X.drop(feature, axis=1)
    presence_of_behavior_encoded_features = [x for x in presence_of_behavior_encoded_X.columns]
    
    #any features not present in the feature encoding map will be added as-is without encoding
    other_features = [x for x in feature_columns if x not in feature_encoding_map]
    other_features_X = df[other_features]
        
    #stitch all sets of features together into one X
    X =  pd.concat([other_features_X.reset_index(drop=True), mixed_numeric_categorical_X.reset_index(drop=True), scalar_encoded_X.reset_index(drop=True), one_hot_encoded_X.reset_index(drop=True), discrete_encoded_X.reset_index(drop=True), presence_of_behavior_encoded_X.reset_index(drop=True)], axis=1)  
    features = other_features + mixed_numeric_categorical_features + scalar_encoded_features + one_hot_encoded_features + discrete_encoded_features + presence_of_behavior_encoded_features
        
    #y is just the target column
    y=np.asarray(df[target_column], dtype="|S20") 
        
    if force_encoded_features != []:
        missing_features = list(set(force_encoded_features) - set(features))
        extra_features = list(set(features) - set(force_encoded_features))
        for feature in missing_features:
            X[feature] = np.zeros(len(X.index))
        for feature in extra_features:
            X = X.drop(feature, axis=1)
        features = force_encoded_features

    return X,y,features

def training_data_statistical_stability_tests(dataset, sample_frac_sizes, feature_columns, feature_encoding_map, target_column, sample_weights, dunno_range, model_function,
        outcome_classes, outcome_class_priors, cross_validate_group_id, n_folds, n_duplicate_runs, do_plotting=False, plot_title='', **model_parameters):
    ''' Run tests to see how statistically limited our sample is (training and X-validation errors) '''
    train_auc_vals = []
    Xvalidate_auc_vals = []
    for sample_frac in sample_frac_sizes:
        duplicate_train_auc_vals = []
        duplicate_Xvalidate_auc_vals = []
        use_n_duplicates = n_duplicate_runs
        if sample_frac < 0.04: use_n_duplicates = 4*n_duplicate_runs
        if sample_frac < 0.08: use_n_duplicates = 2*n_duplicate_runs
        for i in range(use_n_duplicates):   ### run a number of times and average performances to iron out uncertainties
            try:
                #print 'get AUC for sample_frac: ', sample_frac
                frac_dataset = dataset.sample(frac=sample_frac)
                frac_sample_weights = sample_weights.iloc[frac_dataset.index]
                train_model, train_features, train_y_predicted_without_dunno, train_y_predicted_with_dunno, train_y_predicted_probs =\
                    all_data_model(frac_dataset, feature_columns=feature_columns, feature_encoding_map=feature_encoding_map, target_column=target_column,
                            sample_weight=frac_sample_weights, dunno_range=dunno_range, model_function=model_function, **model_parameters)
    
                train_metrics = get_classifier_performance_metrics(outcome_classes, outcome_class_priors, frac_dataset[target_column], train_y_predicted_without_dunno,
                    train_y_predicted_with_dunno, train_y_predicted_probs)
                train_auc = train_metrics['without_dunno']['auc']
                duplicate_train_auc_vals.append(train_auc)
    
    
                Xvalidate_output = cross_validate_model(frac_dataset, sample_weights=frac_sample_weights, feature_columns=feature_columns,
                    feature_encoding_map=feature_encoding_map, target_column=target_column, dunno_range=dunno_range, n_folds=n_folds,
                    outcome_classes=outcome_classes, outcome_class_priors=outcome_class_priors, model_function=model_function, groupid=cross_validate_group_id,
                    **model_parameters)
                Xvalidate_auc = Xvalidate_output['overall_metrics']['without_dunno']['auc']
                print 'For sample_frac: ', sample_frac, ', train AUC: ', train_auc, ', Xvalidate_auc: ', Xvalidate_auc
                duplicate_Xvalidate_auc_vals.append(Xvalidate_auc)
            except:
                print 'Bad statistical fluctuation leading to all samples of same target output. Skip this round.'
        print 'For ', sample_frac, ', average train AUC: ', np.mean(duplicate_train_auc_vals), ', average XV AUC: ', np.mean(duplicate_Xvalidate_auc_vals)
        train_auc_vals.append(np.mean(duplicate_train_auc_vals))
        Xvalidate_auc_vals.append(np.mean(duplicate_Xvalidate_auc_vals))

    if do_plotting:
        plt.figure(figsize=(10,8))
        plt.plot(sample_frac_sizes, train_auc_vals, color='red', label='Training')
        plt.plot(sample_frac_sizes, Xvalidate_auc_vals, color='black', label='Cross validation')
        plt.grid(True)
        plt.xlabel('Fraction of the dataset used', fontsize=20)
        plt.ylabel('Average AUC', fontsize=20)
        plt.title(plot_title, fontsize=22)
        plt.legend(loc='lower right', fontsize=24)
        plt.gca().tick_params(axis='x', labelsize=16)
        plt.gca().tick_params(axis='y', labelsize=16)
        plt.show()

def combine_classifier_performance_metrics(values1, values2, numbers_in_sample_1, numbers_in_sample_2, values1_err=None, values2_err=None):
    ''' 
	Note: this function does not know what the type of metric is. It is the user's responsibility to use this
	function only on valid metrics. This calculation has been verified for recall and precision (so sensitivity
	and specificity are ok). It will not be as accurate for AUC.

	This function could have been written to operate on the output of a single measurement, but it is structured to 
	run on arrays of performance values (values1, values2), and arrays of the numbers in each sample (numbs_in_sample1, ...)
	Each element of the arrays would represent the performance of a different measurement
	

	Note that this combination is for performances at the fixed operating that were used to determine values1 and values2. 
	If these thresholds were optimized independently before the combination it may produce a sub-optimal combined result
	compared to the case where the thresholds of the two algorithms were both floated in the optimization with the combination
	in mind. 

    Assumes that all inputs are numpy arrays of values. If you only want to do this combination for a single
    operating point then arrays can be of size 1.

    values1 and values2 represent arrays of the metric value (recall, or precision)
    numbers_in_sample_1 and numbers_in_sample_2 represent arrays of the number of children in bucket 1 or bucket 2. In the case of autism recall this would be the number
       of children with autism in sample 1 or 2. In the case of autism precision this would be the number of children 
       who the model thinks have autism in sample 1 or 2. If calculating a "real life" precision the "n's" should be 
       chosen to get the correct proportions for the real life hypothesis (any n1 = n2 would be fine for a 50% each real life
       hypothesis)..

    Derivation (done separately on each index of the arrays):
    ~~~~~~~~~
    ### Combining young and old samples: the calculations:
    N_a,c = #  children with autism, correctly diagnosed
    N_a,ic = # children with autism, incorrectly diagnosed
    N_n,c = # children without autism, correctly diagnosed
    N_n,ic = # children without autism, incorrectly diagnosed
    N_a = # children with autism, total = N_a,c + N_a,ic
    N_n = # children without autism, total = N_n,c + N_n,ic
    N_p = # children with a positive diagnosis = N_a,c + N_n,ic
    N_not = # children with a negative diagnosis = N_n,c + N_a,ic
    N_3,x = Same numbers for 3- bin
    N_4,x = Same numbers for 4+ bin, etc ...

    ### Definitions in the 3- bin:
    Recall_3,a = N_3,a,c / (N_3,a,c + N_3,a,ic)
    Recall_3,n = N_3,n,c / (N_3,n,c + N_3,n,ic)
    Precision_3,a = N_3,a,c / (N_3,a,c + N_3,n,ic)
    Precision_3,n = N_3,n,c / (N_3,n,c + N_3,a,ic)

    ### Now calculate definitions in the combined, 3- and 4+ bin (or comparable for young vs old in video)
    **... the definition of combined recall is:**
    Recall_a = (N_3,a,c + N_4,a,c) / (N_3,a,c + N_3,a,ic + N_4,a,c + N_4,a,ic)

    ... now define the denominator as N_tot,a and separate the terms
    Recall_a = (N_3,a,c / N_a) + (N_4,a,c / N_a)

    ... Now use definition N_a = N_3,a * (N_a / N_3,a), and same with 4
    Recall_a = (N_3,a / N_a) * (N_3,a,c / N_3,a) + (N_4,a / N_a) * (N_4,a,c / N_4,a)
    **Recall_a = (N_3,a / N_a) * Recall_3,a + (N_4,a / N_a) * Recall_4,a**
    ... Meaning Recall_a is the weighted average of the recalls in the 3- and the 4+ bins

    **... Similarly, the definition of the combined precision is:**
    Precision_a = (N_3,a,c + N_4,a,c) / (N_3,a,c + N_3,n,ic + N_4,a,c + N_4,n,ic)

    ... Now substitute in terms of number of total with positive diagnosis and separate terms
    Precision_a = (N_3,a,c / N_p) + (N_4,a,c / N_p)

    ... Now do subsitutions of N_p = N_3,p * (N_p / N_3,p)
    Precision_a = (N_3,p / N_p) * (N_3,a,c / N_3,p) + (N_4,p / N_p) * (N_4,a,c / N_4,p)
    ** Precision_a = (N_3,p / N_p) * Precision_3,a + (N_4,p / N_p) * Precision_4,a **

    '''
    assert len(numbers_in_sample_1+numbers_in_sample_2) == len(numbers_in_sample_1)   ### Simple test that n1 and n2 are numpy arrays
    print 'values1: ', values1
    print 'values2: ', values2
    assert len(values1 + values2) == len(values1)
    numbers_in_sample__total = (numbers_in_sample_1 + numbers_in_sample_2).astype(float)
    weights1 = ((numbers_in_sample_1) / numbers_in_sample__total).astype(float)
    weights2 = ((numbers_in_sample_2) / numbers_in_sample__total).astype(float)

    #### weight1 + weight2 = 1.0 by construction, so don't need to divide by weights in average
    weighted_average_metrics = (weights1 * values1) + (weights2 * values2)
    if values1_err is not None and values2_err is not None:
        ### ignore errors on weights: probably small compared to errors on values
        weighted_average_metrics_err = np.sqrt(((weights1*values1_err)**2.) + ((weights2*values2_err)**2.))
    else:
        weighted_average_metrics_err = None
    return weighted_average_metrics, weighted_average_metrics_err



#compute metrics on the predictive power of a multi-class classifer and return them in a dictionary
def get_classifier_performance_metrics(class_names, class_priors, labels, predictions_without_dunno, predictions_with_dunno, prediction_probabilities, weights=None):
    ''' weights optional '''


    #handy function we're going to use a few times in here
    def compute_precision_recall_accuracy(class_names, confusion_matrix):
        precision_per_class = {}
        recall_per_class = {}
        correct = 0
        total = 0
        for class_name in class_names:
    	    class_index = class_names.index(class_name)
    	    correct += confusion_matrix[class_index][class_index]
    	    total += sum(confusion_matrix[class_index])
    	    try:
    	        precision_per_class[class_name] = confusion_matrix[class_index][class_index] / float(sum([line[class_index] for line in confusion_matrix]))
    	    except ZeroDivisionError:
    	        precision_per_class[class_name] = 0.0
    	    try:
    	        recall_per_class[class_name] = confusion_matrix[class_index][class_index] / float(sum(confusion_matrix[class_index])) 
    	    except ZeroDivisionError:
    	        recall_per_class[class_name] = 0.0
    	try:
    	    accuracy = float(correct) / float(total)
    	except ZeroDivisionError:
    	    accuracy = 0.0
        return precision_per_class, recall_per_class, accuracy


    #handy function we're going to use a few times here
    def apply_priors_to_confusion_matrix(matrix, priors):
        new_matrix = matrix.copy()
        matrix_total = float(sum(sum(matrix)))
        for i in range(0, len(priors)):
            class_prior = priors[i]
            class_proportion = float(sum(matrix[i]))/matrix_total if matrix_total>0.0 else 0.0
            class_multiplier = (class_prior / class_proportion) if class_proportion!=0 else 1 
            new_matrix[i] = np.array([100*value*class_multiplier for value in new_matrix[i]])
        return new_matrix
           

    #
    # metrics related to dataset profile 
    #
    
    number_samples = len(labels)

    #compute number of samples for every class
    samples_per_class = {}

    for class_name in class_names:
        samples_per_class[class_name] = len([x for x in labels if x==class_name])
    

    #
    # metrics related to classification excluding dunno class logic 
    #
	positive_probabilities = [x[0] for x in prediction_probabilities]
    auc_without_dunno = metrics.roc_auc_score([x==class_names[0] for x in labels], positive_probabilities, sample_weight=weights)
    dataset_confusion_without_dunno = confusion_matrix_0p18(labels, predictions_without_dunno, labels=class_names, sample_weight=weights)
    dataset_precision_per_class_without_dunno, dataset_recall_per_class_without_dunno, dataset_accuracy_without_dunno = compute_precision_recall_accuracy(class_names, dataset_confusion_without_dunno)
        
    reallife_confusion_without_dunno = apply_priors_to_confusion_matrix(dataset_confusion_without_dunno, class_priors)
    reallife_precision_per_class_without_dunno, reallife_recall_per_class_without_dunno, reallife_accuracy_without_dunno = compute_precision_recall_accuracy(class_names, reallife_confusion_without_dunno)
    
    
    #
    # metrics related to classification including dunno class logic 
    # 
       
    #dataset_confusion_with_dunno = metrics.confusion_matrix(labels, predictions_with_dunno, class_names+['dunno'])
    try:
        dataset_confusion_with_dunno = confusion_matrix_0p18(labels, predictions_with_dunno, class_names+['dunno'], sample_weight=weights)
        dataset_precision_per_class_with_dunno, dataset_recall_per_class_with_dunno, dataset_accuracy_with_dunno = compute_precision_recall_accuracy(class_names, dataset_confusion_with_dunno)

        reallife_confusion_with_dunno = apply_priors_to_confusion_matrix(dataset_confusion_with_dunno, class_priors)
        reallife_precision_per_class_with_dunno, reallife_recall_per_class_with_dunno, reallife_accuracy_with_dunno = compute_precision_recall_accuracy(class_names, reallife_confusion_with_dunno)
    except Exception as msg:   ### usually because of so broad a dunno range that confusion matrix is not defined
        #print 'Getting classifier performance metrics with dunno classified failed with message ', msg
        dataset_confusion_with_dunno = None
        dataset_precision_per_class_with_dunno = {class_name: np.nan for class_name in class_names}
        dataset_recall_per_class_with_dunno = {class_name: np.nan for class_name in class_names}
        dataset_accuracy_with_dunno = np.nan
        reallife_confusion_with_dunno = None
        reallife_precision_per_class_with_dunno = {class_name: np.nan for class_name in class_names}
        reallife_recall_per_class_with_dunno = {class_name: np.nan for class_name in class_names}
        reallife_accuracy_with_dunno = {class_name: np.nan for class_name in class_names}
        dataset_classification_rate = 0.
        reallife_classification_rate = 0.
 
   
    #create a list of labels, predictions and scores excluding the unclassified cases
    z = zip(labels.tolist(), predictions_with_dunno.tolist(), positive_probabilities)
    labels_where_classified = [x[0] for x in z if x[1]!="dunno"]
    predictions_where_classified = [x[1] for x in z if x[1]!="dunno"]
    probabilities_where_classified = [x[2] for x in z if x[1]!="dunno"]
    if weights is not None:
        z = zip(weights.tolist(), predictions_with_dunno.tolist())
        weights_where_classified = [x[0] for x in z if x[1]!="dunno"]
    else:
        weights_where_classified = None
    
    #then compute some metrics over those
    #dataset_confusion_where_classified = metrics.confusion_matrix(labels_where_classified, predictions_where_classified, class_names)
    try:
        dataset_confusion_where_classified = confusion_matrix_0p18(labels_where_classified, predictions_where_classified, class_names, sample_weight=weights_where_classified)
        dataset_precision_per_class_where_classified, dataset_recall_per_class_where_classified, dataset_accuracy_where_classified = compute_precision_recall_accuracy(class_names, dataset_confusion_where_classified) 
        reallife_confusion_where_classified = apply_priors_to_confusion_matrix(dataset_confusion_where_classified, class_priors)
        reallife_precision_per_class_where_classified, reallife_recall_per_class_where_classified, reallife_accuracy_where_classified = compute_precision_recall_accuracy(class_names, reallife_confusion_where_classified) 
        dataset_classification_rate = float(sum(sum(dataset_confusion_where_classified))) / float(sum(sum(dataset_confusion_with_dunno)))
        reallife_classification_rate = float(sum(sum(reallife_confusion_where_classified))) / float(sum(sum(reallife_confusion_with_dunno)))
    except Exception as msg:
        #print 'Getting classifier performance metrics where classified failed with message ', msg
        dataset_confusion_where_classified = None
        dataset_precision_per_class_where_classified = {class_name: np.nan for class_name in class_names}
        dataset_recall_per_class_where_classified = {class_name: np.nan for class_name in class_names}
        dataset_accuracy_where_classified = np.nan
        reallife_confusion_where_classified = None
        reallife_precision_per_class_where_classified = {class_name: np.nan for class_name in class_names}
        reallife_recall_per_class_where_classified = {class_name: np.nan for class_name in class_names}
        reallife_accuracy_where_classified = {class_name: np.nan for class_name in class_names}
        dataset_classification_rate = 0.
        reallife_classification_rate = 0.

    try:     #### May fail when others succeed due to roc_auc_score demand of presence of both classes
        auc_where_classified = metrics.roc_auc_score([label==class_names[0] for label in labels_where_classified], probabilities_where_classified, sample_weight=weights_where_classified)
    except:
        auc_where_classified = np.nan

    output = {
        'number_samples':number_samples, 
        'samples_per_class':samples_per_class,
        'without_dunno': {
            'auc': auc_without_dunno,
            'dataset_accuracy':dataset_accuracy_without_dunno,
            'reallife_accuracy':reallife_accuracy_without_dunno,
            'dataset_precision_per_class':dataset_precision_per_class_without_dunno,
            'reallife_precision_per_class':reallife_precision_per_class_without_dunno,
            'dataset_recall_per_class':dataset_recall_per_class_without_dunno, 
            'reallife_recall_per_class':reallife_recall_per_class_without_dunno, 
            'dataset_confusion': dataset_confusion_without_dunno,
        },
        'with_dunno': {
            'auc': auc_where_classified,
        	'dataset_classification_rate': dataset_classification_rate,
        	'reallife_classification_rate': reallife_classification_rate,
            'dataset_accuracy_where_classified': dataset_accuracy_where_classified,
            'reallife_accuracy_where_classified': reallife_accuracy_where_classified,
            'dataset_precision_per_class':dataset_precision_per_class_with_dunno,
            'reallife_precision_per_class':reallife_precision_per_class_with_dunno,
            'dataset_recall_per_class':dataset_recall_per_class_with_dunno, 
            'reallife_recall_per_class':reallife_recall_per_class_with_dunno, 
            'dataset_precision_per_class_where_classified':dataset_precision_per_class_where_classified,
            'reallife_precision_per_class_where_classified':reallife_precision_per_class_where_classified,
            'dataset_recall_per_class_where_classified':dataset_recall_per_class_where_classified, 
            'reallife_recall_per_class_where_classified':reallife_recall_per_class_where_classified, 
        	'dataset_confusion': dataset_confusion_with_dunno,
        }

    }
       
    return output

#pretty printing of outputs that you got by calling get_multi_classifier_performance_metrics
def print_classifier_performance_metrics(class_names, metrics):

    print("\n\nDATASET PROFILE:\n\n")
    
    print("samples\t\t"+str(metrics['number_samples']))
    for class_name in class_names:
        print(class_name+
        " samples\t"+
        str(metrics['samples_per_class'][class_name])+
        "\t["+
        str(100.0*float(metrics['samples_per_class'][class_name]) / float(metrics['number_samples'])) 
        +"%]"
        )
    
    print("\n\nPERFORMANCE BEFORE DUNNO LOGIC:\n\n")

    print("AUC\t\t"+str(metrics['without_dunno']['auc']))
    print("accuracy [Dataset]\t\t"+str(metrics['without_dunno']['dataset_accuracy']))
    print("accuracy [Reallife]\t\t"+str(metrics['without_dunno']['reallife_accuracy']))

    print("\n")
    for class_name in class_names:
        print(class_name+" precision [Dataset]\t"+str(metrics['without_dunno']['dataset_precision_per_class'][class_name]))
        print(class_name+" precision [Reallife]\t"+str(metrics['without_dunno']['reallife_precision_per_class'][class_name]))
        print(class_name+" recall\t"+str(metrics['without_dunno']['dataset_recall_per_class'][class_name]))
    print("\n")
    print("Confusion Matrix:")
    print(metrics['without_dunno']['dataset_confusion'])

    print("\n\nPERFORMANCE INCLUDING DUNNO LOGIC:\n\n")

    print("classifcation rate [Dataset]\t"+str(metrics['with_dunno']['dataset_classification_rate']))    
    print("classifcation rate [Reallife]\t"+str(metrics['with_dunno']['reallife_classification_rate']))    
    print("accuracy where classified [Dataset]\t"+str(metrics['with_dunno']['dataset_accuracy_where_classified']))
    print("accuracy where classified [Reallife]\t"+str(metrics['with_dunno']['reallife_accuracy_where_classified']))

    print("\n")
    for class_name in class_names:
        print(class_name+" precision [Dataset]\t"+str(metrics['with_dunno']['dataset_precision_per_class'][class_name]))
        print(class_name+" precision [Reallife]\t"+str(metrics['with_dunno']['reallife_precision_per_class'][class_name]))
        print(class_name+" recall\t"+str(metrics['with_dunno']['dataset_recall_per_class'][class_name]))
    print("\n")
    for class_name in class_names:
        print(class_name+" precision where classified [Dataset]\t"+str(metrics['with_dunno']['dataset_precision_per_class_where_classified'][class_name]))
        print(class_name+" precision where classified [Reallife]\t"+str(metrics['with_dunno']['reallife_precision_per_class_where_classified'][class_name]))
        print(class_name+" recall where classified\t"+str(metrics['with_dunno']['dataset_recall_per_class_where_classified'][class_name]))
    print("\n")
    print("Confusion Matrix:")
    print(metrics['with_dunno']['dataset_confusion'])
    

#returns a sorted list of (important feature, importance value) pairs for a passed model
def get_important_features(model, feature_names, relative_weight_cutoff=0.01):
    sorted_feature_importances = sorted(zip(feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True)
    max_feature_importance = sorted_feature_importances[0][1]
    min_feature_importance = relative_weight_cutoff*max_feature_importance
    trimmed_sorted_feature_importances = [x for x in sorted_feature_importances if x[1]>min_feature_importance]
    return trimmed_sorted_feature_importances

#name says it all
def dedup_list(mylist):
    newlist = []
    for i in mylist:
        if i not in newlist:
            newlist.append(i)
    return newlist

#given the output of get_important_features(), returns the n best features to keep 
#ignoring anything after a certain suffix char in the feature name, and excluding certain features
def get_best_features(important_features, number_of_features_to_keep, identifiers_for_suffixes_to_ignore, features_to_skip):
    ''' gets best features among features that are not skipped. Encoding may lead to variable names
    like 'feature=3' or 'feature_behavior_present'. identifiers_for_suffixes_to_ignore is a list
    of strings. If any of these strings is contained in a feature name, function will ignore that string
    and any suffix following it '''

    candidate_features = [x[0] for x in important_features]

    #trim everything after 'suffix_to_ignore' in every feature
    for string_for_suffix_to_ignore in identifiers_for_suffixes_to_ignore:
        candidate_features = [x.split(string_for_suffix_to_ignore)[0] for x in candidate_features]
    
    #skip features to skip
    candidate_features = [x for x in candidate_features if x not in features_to_skip]
    
    #dedup
    candidate_features = dedup_list(candidate_features)
  
    #return top N candidate features
    return candidate_features[0:number_of_features_to_keep]

def get_modeled_results(model, X, dunno_range):
    ''' Helper function to extract modeled results for
    all_data_model_withAlternates function below '''
    y_predicted_without_dunno = model.predict(X)
    y_predicted_probs = model.predict_proba(X)
    y_predicted_with_dunno = np.array(y_predicted_without_dunno, copy=True)
    if dunno_range:
        for i in range(0,len(y_predicted_with_dunno)):
            if (dunno_range[0] < y_predicted_probs[i][0] < dunno_range[1]):
                y_predicted_with_dunno[i] = "dunno"
    return y_predicted_without_dunno, y_predicted_with_dunno, y_predicted_probs


import collections



#trains a model using the entire dataset passed
def all_data_model(dataset, feature_columns, feature_encoding_map, target_column, sample_weight, dunno_range, model_function, **model_parameters):


    X,y,features = prepare_data_for_modeling(dataset, feature_columns, feature_encoding_map, target_column)
    
    model = model_function(**model_parameters)
    if (sample_weight is not None):
    	model.fit(X=X, y=y, sample_weight=sample_weight.values)
    else:
    	model.fit(X=X, y=y)
    
    y_predicted_without_dunno = model.predict(X)
    y_predicted_probs = model.predict_proba(X) 
    

    #replace predicted class with "dunno" class where appropriate
    y_predicted_with_dunno = np.array(y_predicted_without_dunno, copy=True)
    if dunno_range:
        for i in range(0,len(y_predicted_with_dunno)):  
            if (dunno_range[0] < y_predicted_probs[i][0] < dunno_range[1]):
                y_predicted_with_dunno[i] = "dunno"


    return model, features, y_predicted_without_dunno, y_predicted_with_dunno, y_predicted_probs

def get_stratified_labeled_KFolds(myDF, n_folds, target_column='diagnosis', groupKey=None):
    ''' Mixes the functionality of stratified k-fold and labeled k-fold
    Will split into n folds with approximately equal amounts of each dignosis.
    Groups, as defined by rows with common values of groupKey, will be preserved when
    doing this. If groupKey is left as None this functionality is ignored
    and the function reduces to simple StratifiedKFolds.

    Returns: an n-element list of training groups and an n-element list of validate
    groups. These groups are each lists of positions of the rows that should be selected
    in that particular fold '''

    #myDF[target_column+'encoding'] = myDF[target_column].apply(lambda x : 1. if x == 'autism' else 0.)
    if groupKey is None:
        ### no grouping to do
        avgByGroupDF = myDF
        #valsForFolding = myDF[target_column+'encoding'].values
        #### This just reduces to the non labeled KFold case
        valsForFolding = myDF[target_column].values
    else:
        #avgByGroupDF = myDF.groupby(groupKey).apply(np.mean)
        #valsForFolding = avgByGroupDF[target_column+'encoding'].values
        
        #### If more than one row is in group, hopefully they all have the same
        #### value of the target column. If not then take the most common value
        #### (mode) as the one to use for the group.
        #### Note: the [0][0] is actually needed to retrieve the mode from
        #### the weird object that scipy stats returns
        avgByGroupDF = myDF.groupby(groupKey).agg(lambda x: stats.mode(x)[0][0])
        valsForFolding = avgByGroupDF[target_column].values
    cross_validation_folds = cross_validation.StratifiedKFold(n_folds=n_folds, y=valsForFolding, shuffle=True)
    #print 'cross validation folds: ', cross_validation_folds

    trainingGroups, validateGroups = [], []

    ### Cross validation folds tell us the numpy row index values (*not the pandas index*)
    ### for the training and validation data for each fold 
    ### of the rows in the underlying 2d numpy array of the avgByGroupDF dataframe
    ### .... next need to convert this back to the assocaited numpy index values for the 
    ### non-grouped dataframe (myDF)
    ### If not running with groups then avgByGroupDF is the same as myDF so 
    ### this will just return the original values
    for train,validate in cross_validation_folds:

        if groupKey is None:   ### no grouping, so no need to map back to original DF
            trainingGroups.append(train)
            validateGroups.append(validate)
        else:   ### Need to map backto original DF
            trainingIDs = avgByGroupDF.index.values[train]
            validateIDs = avgByGroupDF.index.values[validate]
            origTrainingPositions = np.where(myDF[groupKey].isin(trainingIDs) == True)[0]
            origValidatePositions = np.where(myDF[groupKey].isin(validateIDs) == True)[0]
            trainingGroups.append(origTrainingPositions)
            validateGroups.append(origValidatePositions)

    return trainingGroups, validateGroups

def cross_validate_model(dataset, sample_weights, feature_columns, feature_encoding_map,
        target_column, dunno_range, n_folds, outcome_classes, outcome_class_priors, model_function,
        groupid=None, validation_weights=None, **model_parameters):
    ''' sample_weights is for the actual training.

    validation_weights is optional in case you want certain events to carry more importance in the 
    cross validation (need not match sample_weights) '''

    #we will return this
    metrics = {'fold_metrics':[], 'fold_important_features':[], 'overall_metrics':[]}
    df_to_use = cp.deepcopy(dataset)
    if sample_weights is not None:
        df_to_use['sample_weights'] = sample_weights

    X,y,features = prepare_data_for_modeling(dataset, feature_columns, feature_encoding_map, target_column)
    metrics['features'] = features

    
    #prepare the dataset for modeling
    #X,y,features = prepare_data_for_modeling(dataset, feature_columns, feature_encoding_map, target_column)

    #split the dataset into n_folds cross validation folds
    #cross_validation_folds = cross_validation.StratifiedKFold(n_folds=n_folds, y=df_to_use[target_column].values, shuffle=True)
    trainPositionsGroups, validatePositionsGroups = get_stratified_labeled_KFolds(df_to_use, n_folds=n_folds, target_column=target_column, groupKey=groupid)

    #these are going to be overall prediction lists across folds
    overall_y_real = []
    overall_y_predicted_without_dunno = []
    overall_y_predicted_with_dunno = []
    overall_y_predicted_probs = []
    overall_validation_weights = None if validation_weights is None else []

    #indices_with_correct_Xvalid_results = np.array([])
    correctly_predicted_sample_indices = np.array([])
    
    #handle folds one at  at ime
    fold_num = 0
    for trainPositions, validatePositions in zip(trainPositionsGroups, validatePositionsGroups):
        fold_num += 1

        X_train = X.iloc[trainPositions]
        X_validate = X.iloc[validatePositions]
        y_train = pd.Series(y).iloc[trainPositions]
        y_validate = pd.Series(y).iloc[validatePositions]
        sample_weight_train = sample_weights.iloc[trainPositions]
        weights_validate = None if validation_weights is None else validation_weights.iloc[validatePositions]

        
        model = model_function(**model_parameters)
        if sample_weights is not None:
            model.fit(X=X_train, y=y_train, sample_weight=sample_weight_train.values)
        else:
            model.fit(X=X_train, y=y_train)
        
        y_predicted_without_dunno = model.predict(X=X_validate)
        y_predicted_probs = model.predict_proba(X=X_validate)

        #replace predicted class with "dunno" class where appropriate
        y_predicted_with_dunno = np.array(y_predicted_without_dunno, copy=True)
        if dunno_range:
            for i in range(0,len(y_predicted_with_dunno)):  
                if (dunno_range[0] < y_predicted_probs[i][0] < dunno_range[1]):
                    y_predicted_with_dunno[i] = "dunno"
         
        #maintain overall prediction lists across folds   
        overall_y_real.extend(y_validate.values)
        overall_y_predicted_without_dunno.extend(y_predicted_without_dunno)
        overall_y_predicted_with_dunno.extend(y_predicted_with_dunno)
        overall_y_predicted_probs.extend(y_predicted_probs)
        if weights_validate is not None:
            overall_validation_weights.extend(weights_validate)
        
        values_match = y_validate == y_predicted_without_dunno
        indices_that_match = values_match[values_match==True].index

        correctly_predicted_sample_indices = np.concatenate([correctly_predicted_sample_indices, indices_that_match])
        
        #log metrics for this fold
        metrics['fold_metrics'].append( get_classifier_performance_metrics(outcome_classes, outcome_class_priors, y_validate, y_predicted_without_dunno, y_predicted_with_dunno, y_predicted_probs, weights_validate) )
        metrics['fold_important_features'].append ( get_important_features(model, features, 0.01) )


    #log overall metrics across folds
    validation_weights_arr = None if overall_validation_weights is None else np.array(overall_validation_weights)
    metrics['overall_metrics'] = get_classifier_performance_metrics(outcome_classes, outcome_class_priors, np.array(overall_y_real), np.array(overall_y_predicted_without_dunno), np.array(overall_y_predicted_with_dunno), np.array(overall_y_predicted_probs), validation_weights_arr)
    metrics['correctly_predicted_sample_indices'] = correctly_predicted_sample_indices
    

    return metrics




#performs general purpose bootstrapping on a dataset applying given function each time and aaverageing reported metrics back to caller
def bootstrap(dataset, number_of_tries, sampling_function_per_try, ml_function_per_try, return_errs=False, verbose=False ):
    metrics_per_try = []
    for try_index in range(0, number_of_tries):
        if verbose:
            print 'On try_index ', try_index, ' out of ', number_of_tries
        #sample dataset before every try
        dataset_for_this_try = sampling_function_per_try(dataset)
        
        #run the ml function and collect reported metrics
        metrics_per_try.append( ml_function_per_try(dataset_for_this_try) )
        
    #average out the metrics in the (nested) metrics structures that we got back from every try
    if return_errs:
        average_metrics, average_metrics_err = deep_average_with_stat_errs(metrics_per_try)
        return average_metrics, average_metrics_err
    else:
        average_metrics = deep_average(metrics_per_try)
        return average_metrics
    return average_metrics


#SAVING AND LOADING PREDICTIVE MODELS
def save_model( model, class_names, dunno_range, features, feature_columns, feature_encoding_map, outcome_column, data_prep_function, apply_function, filename):

    data_prep_function_string = marshal.dumps(data_prep_function.func_code)
    apply_function_string = marshal.dumps(apply_function.func_code)

    pickle.dump( {'model':model, 'class_names': class_names, 'dunno_range':dunno_range, 'features':features, 'feature_columns':feature_columns,
		'feature_encoding_map': feature_encoding_map, 'target':outcome_column, 'data_prep_function':data_prep_function_string, 'apply_function':apply_function_string}, open( filename, "wb" ) )

def load_model( filename):
    model_structure = pickle.load(open( filename, "rb" ) )
    data_prep_function_key = 'data_prep_function'
    apply_function_key = 'apply_function'

    model_structure[data_prep_function_key] = types.FunctionType(marshal.loads(model_structure[data_prep_function_key]), globals(), data_prep_function_key)
    model_structure[apply_function_key] = types.FunctionType(marshal.loads(model_structure[apply_function_key]), globals(), apply_function_key)

    return model_structure

#given a bunch of variables, each in a list of possible ranges, returns tuples for every possible combination of variable values
#   for example, given [ ['a','b'], [1,2], ['x','y'] ]
#   it returns [ ('a',1,'x'), ('a',1,'y'), ('a',2,'x'), ('a',2,'y'), ('b',1,'x') ... ]
def get_combinations(list_of_lists):
    return list(itertools.product(*list_of_lists))

#performs a grid search over a modeling_function using different param_combinations each time, and collecting the outputs of a reporting_fuction in successive runs
def grid_search(modeling_function, param_combinations, reporting_function, verbose=False):
    final_report = []
    n_total_combs = len(param_combinations)
    n_so_far = 0
    for param_combination in param_combinations:
        n_so_far += 1
        if verbose:
            print 'Starting grid search parameter combination ', param_combination
            print 'This is combination ', n_so_far, ' of ', n_total_combs
        metrics = modeling_function(param_combination)	
        report = list(param_combination)
        report.extend( reporting_function(param_combination, metrics) )
        final_report.append(report)
    return final_report
    
        


def injectLoss(inputDF, lossSpecifications, missingValue=999, mode='duplicate', probKey='probability', scaleToOverwrite=False, exactMatch=False):
    ''' Function to inject loss simulating unknowable results in an imperfect survey

    Example epected input format:
        lossSpecifications = {'desc': '10pLoss, 'instructions': [{'qType': 'instrument_module2_a', 'probability': 0.1},
                                                                 {'qType': 'instrument_module2_b', 'probability': 0.1},
                                                                 ...]}
    Method: for every event, throws random number between 0 and 1 for each element of instructions
    If result is less than probability, all questions matching qType description will have their values
    Reverted to a missing value

    'missingValue': either a value to replace with, or a 'rand', which represents a random integer
    from [0,1,2,3,4]

    mode can be 'duplicate' or 'overwrite'
    ... duplicate means to make a copy of the row with the values missing and append it
    ...... If scaleToOverwrite is set to True then the probability will be scaled up so that 
    ...... The final number of missing values is the same as expected from the overwrite option
    .......... Caveat: obviously since the max probability is 1.0 no probability value over 0.5
    .......... can work with this option unless multiple copies of the missing events are used
    .......... and in that case it becomes ambiguous when multiple missing values with tdiferent probabilities
    .......... are specified in the instructions.
    ... overwrite means overwrite the current row with the new one that has the missing values

    Returns data frame rebuilt with appropriate missing values '''

    desc = lossSpecifications['desc']
    instructions = lossSpecifications['instructions']
    if instructions is None:
        return inputDF

    #### What is probability of having a row lost??
    pGivenRowNotLost = 1.
    for instruct in instructions:
        pGivenRowNotLost *= 1. - max(instruct[probKey], 0.9999999)
    pGivenRowLost = 1. - pGivenRowNotLost

    nRows = len(inputDF.index)
    outputDF = pd.DataFrame()
    colsAlreadyDone = []
    colsToReset = []
    resetDF = cp.deepcopy(inputDF)
    resetDF['colsToReset'] = [[]]*len(resetDF.index)
    def markColsForReset(row, colsToReset):
        return row['colsToReset'].append(colsToReset)

    #### Loop over the instructions for loss types to apply
    for instr in instructions:
        instrPLoss = instr[probKey]

        if scaleToOverwrite:
            ### Correct for the duplication factor
            assert mode == 'duplicate'
            instrPLoss = instrPLoss / (1. - min(pGivenRowLost, 0.5))
            if instrPLoss > 1.:
                print 'Warning, hit physical ceiling'
                instrPLoss = 1.   #### hit physical ceiling
            #### Now instrPLoss has been scaled to 
            #### value that will make the output
            #### missing rate equal this probability
            #### after non-missing duplicates are factored in
        #applyLossToTheseCols = [ele for ele in inputDF.columns if instr['qType'] in ele and instr['except'] not in ele]
        matchCheck = lambda x : instr['qType'] in ele
        if exactMatch:
              matchCheck = lambda x : instr['qType'] == ele

        if 'except' in instr.keys(): applyLossToTheseCols = [ele for ele in inputDF.columns if matchCheck(ele) and instr['except'] not in ele]
        else: applyLossToTheseCols = [ele for ele in inputDF.columns if matchCheck(ele)]

        ### Check that none of the new columns have already had instructions
        overlapCols = np.intersect1d(colsAlreadyDone, applyLossToTheseCols)
        if len(overlapCols) != 0:
            print 'Error, ', overlapCols, ' have conflicting instructions. Abort.'
            raise ValueError
        colsAlreadyDone += applyLossToTheseCols

        #### get which rows need to have this subset of columns reset
        rowsToReset = (np.random.rand(nRows) < instrPLoss)
        #print 'rowsToReset: ', rowsToReset

        #### expand the list of columns that do need a reset for each row:
        resetDF['colsToReset'] = [list(curContent) + applyLossToTheseCols if doThisRow else list(curContent)\
                  for curContent, doThisRow in zip(resetDF['colsToReset'].values, rowsToReset)]

    #### Define the pandas operation that will do the reset
    def doResets(row):
        outRow = row
        #print 'cols to reset: ', row['colsToReset']
        for key in row['colsToReset']:
            #print 'missingValue: ', missingValue
            if type(missingValue) == str and missingValue == 'random':
                randValToUse = int(5.*np.random.rand(1)[0])
                outRow[key] = randValToUse
            else:
                #print 'set outRow of ', key, ' to missing Value ', missingValue
                outRow[key] = missingValue
        return outRow

    ### inject some tracking information to enable reconstruction of original later
    resetDF['original_index'] = resetDF.index
    resetDF['status'] = ['original']*len(resetDF.index)
    #### Now actually apply the resets
    if mode == 'overwrite':
        outDF = resetDF.apply(doResets, axis=1)

    if mode == 'duplicate':
        ### In this case we keep a copy of the original rows in the DF
        appendDF = resetDF[np.array([ele != [] for ele in resetDF['colsToReset']])]
        appendDF = appendDF.apply(doResets, axis=1)
        appendDF['status'] = ['duplicate']*len(appendDF.index)
        outDF = resetDF.append([appendDF], ignore_index=True)
    #print 'outDF: ', outDF
    return outDF

def inject_proportional_loss_when_presence_encoding(inputDF, outcome_key='outcome', instructions=None, missing_value='missing', prior_autism_frac=None, module=None, validation=True):
    ''' 
    instructions: if None will be filled with expected default values. Format should be
    a list of dictionaries where each entry tells the association of the feature (whether
    presence means autism, non-autism, or neutral). 
    
    Example:
    instructions = [{'feature': 'insturment_module1_a1', 'presence_means': 'autism'}, {'feature': 'insturment_module1_a2', 'presence_means': 'not'},
                    {'feature': 'insturment_module1_a3', 'presence_means': 'neutral'}]
    prior_autism_frac: if you want proportionality after reweighting priors then specify prior_autism_frac
    module : if you want to only inject in features of a given module then add this requirement
    Intended to inject loss for autism cases and not autism cases separately, with the goal of
    making it so that the non-presence of a feature is not used to make any important decisions in the
    trees. 
    
    To accomplish this, each feature is considered to be either one where the presence implies the child
    is more likely to be autistic, or less likely to be autistic. If more likely, then the absence of the feature
    is likely to be interpreted as a reduction in the autism probability. To compensate for this missing values
    should be injected in real autism cases until the fraction is in balance with the non autism cases. '''


    def get_frac_injection(no_presence_df, presence_df, outcome_key, presence_means, autism_scaling_factor):
        #### How much do we need to inject to achieve balance in the non-presence category??
        n_not_no_presence = float(len(no_presence_df[no_presence_df[outcome_key]=='not'].index))
        n_autism_no_presence = float(len(no_presence_df[no_presence_df[outcome_key]=='autism'].index))
        n_not_presence = float(len(presence_df[presence_df[outcome_key]=='not'].index))
        n_autism_presence = float(len(presence_df[presence_df[outcome_key]=='autism'].index))
        if autism_scaling_factor is not None:
            n_autism_no_presence *= autism_scaling_factor
            n_autism_presence *= autism_scaling_factor


        if presence_means == 'autism':   ### inject into autism to make frac as large as not in non-presence category
            frac_injection = (n_not_no_presence -  n_autism_no_presence) / (n_autism_no_presence + n_autism_presence)
            #if autism_scaling_factor is not None:  ### correct values to ensure balancing is to correct priors
            #    #print 'Fraction to inject with full accounting for weighting is ', frac_injection
            #    #print 'Injection will be before weighting to desired prior of ', prior_autism_frac, ', so roll this out'
            #    frac_injection /= autism_scaling_factor
            #    #print 'Corrected for weightings, frac_injection is ', frac_injection
        elif presence_means == 'not':   ### inject into not to make frac as large as autism in non-presence category
            frac_injection = (n_autism_no_presence - n_not_no_presence) / (n_not_no_presence + n_not_presence)
        else:
            print 'type of injection = ', presence_means, ' not understood'
            raise ValueError


        if frac_injection < 0:
            print 'Warning: frac_injection < 0, meaning selection feature is not enriching correct outcome. Skip injection.'
            return None
        else:
            return frac_injection


    feature_columns = [instruction['feature'] for instruction in instructions if instruction['feature'] in inputDF.columns]
    feature_encoding_map = {feature: 'presence_of_behavior' for feature in feature_columns}
    #print 'feature_columns: ', feature_columns
    #print 'feature_encoding_map: ', feature_encoding_map

    ### transform the data into presence encoded results
    encoded_df, _, _ = prepare_data_for_modeling(inputDF, feature_columns, feature_encoding_map, target_column=outcome_key)
    ### The columns will have a '_behavior_present' suffix. Remove this.
    new_cols = [col[:-len('_behavior_present')] if col.endswith('_behavior_present') else col for col in encoded_df.columns]
    encoded_df.columns = new_cols
    encoded_df[outcome_key] = cp.deepcopy(inputDF[outcome_key])

    autism_prior_scaling_factor = None
    if prior_autism_frac is not None:  ### correct values to ensure balancing is to correct priors
        if prior_autism_frac > 0.99999 or prior_autism_frac < 0.000001:
            print 'Error, prior_autism_frac: ', prior_autism_frac, ' not understood'
            raise ValueError
        n_autism_tot = float(len(inputDF[inputDF[outcome_key]=='autism'].index))
        n_not_tot = float(len(inputDF.index) - n_autism_tot)
        autism_prior_scaling_factor = prior_autism_frac * n_not_tot / (n_autism_tot * (1. - prior_autism_frac))

        #print 'prior_autism_frac of ', prior_autism_frac, ' is desired.'
        #print 'Initial n_autism: ', n_autism_tot, ', n_not: ', n_not_tot
        #print 'Scaling factor is ', autism_prior_scaling_factor

    ### First determine what our loss instructions should be
    autism_loss_instructions = {'desc': 'autism_loss_instructions', 'instructions': []}
    not_loss_instructions = {'desc': 'not_loss_instructions', 'instructions': []}
    suspicious_features = []
    for instruct in instructions:
        feature = instruct['feature']
        if module is not None:
            if 'instrument_'+str(module) not in feature: continue
        if feature not in encoded_df.columns:
#            print 'Warning, feature ', feature, ' not understood in dataframe columns: ', encoded_df.columns
#            print 'Skip it'
            continue
        presence_means = instruct['presence_means']
        if presence_means == 'neutral': continue
        elif presence_means == 'autism':
            ### May need to inject missing values into 'not' in order to achieve balance
            no_presence_df = encoded_df[encoded_df[feature]==0][[feature, outcome_key]]
            presence_df = encoded_df[encoded_df[feature]==1][[feature, outcome_key]]
            #n_not = float(len(no_presence_df[no_presence_df[outcome_key]=='not'].index))
            #n_autism = float(len(no_presence_df[no_presence_df[outcome_key]=='autism'].index))
            needed_frac_injection = get_frac_injection(no_presence_df, presence_df, outcome_key, presence_means, autism_prior_scaling_factor)
            if needed_frac_injection is None:
                suspicious_features.append(feature)
            else:
                autism_loss_instructions['instructions'].append({'qType': feature, 'probability': needed_frac_injection})
                if needed_frac_injection>0.5:
                    suspicious_features.append(feature)
            #print 'for feature: ', feature, ', needed_frac_injection: ', needed_frac_injection
        elif presence_means == 'not':
            ### May need to inject missing values into 'autism' in order to achieve balance
            no_presence_df = encoded_df[encoded_df[feature]==0][[feature, outcome_key]]
            presence_df = encoded_df[encoded_df[feature]==1][[feature, outcome_key]]
            needed_frac_injection = get_frac_injection(no_presence_df, presence_df, outcome_key, presence_means, autism_prior_scaling_factor)

            if needed_frac_injection is None:
                suspicious_features.append(feature)
            else:
                not_loss_instructions['instructions'].append({'qType': feature, 'probability': needed_frac_injection})
                if needed_frac_injection>0.5:
                    suspicious_features.append(feature)
        else:
            print 'Error, instructions ', instruct, ' not understood. Abort.'
            return -1

    ### Now apply our loss instructions
    #print 'Got loss instructions for autism and not DFs separately'
    #print 'autism instructions: ', autism_loss_instructions
    #print 'not instructions: ', not_loss_instructions
    autism_df = inputDF[inputDF[outcome_key]=='autism']
    not_df = inputDF[inputDF[outcome_key]=='not']
    if len(autism_loss_instructions['instructions']) > 0:   ## apply our loss instructions
        autism_df = injectLoss(autism_df, autism_loss_instructions, missingValue=missing_value, mode='duplicate',
             probKey='probability', scaleToOverwrite=True, exactMatch=True)
    if len(not_loss_instructions['instructions']) > 0:    ## apply our loss instructions
        not_df = injectLoss(not_df, not_loss_instructions, missingValue=missing_value, mode='duplicate',
             probKey='probability', scaleToOverwrite=True, exactMatch=True)

    ### Now merge the results and re-shuffle to avoid grouping by autism / not
    output_df = autism_df.append([not_df], ignore_index=True)
    output_df = (output_df.reindex(np.random.permutation(output_df.index))).reset_index()


    if validation:
        do_proportional_injection_sanity_checks(encoded_df, output_df, instructions, feature_columns, suspicious_features, feature_encoding_map, outcome_key, autism_prior_scaling_factor, module)

    return output_df

def do_proportional_injection_sanity_checks(encoded_df, output_df, instructions, feature_columns, suspicious_features, feature_encoding_map, outcome_key, autism_prior_scaling_factor, module):
    autism_scale_factor = 1. if autism_prior_scaling_factor is None else autism_prior_scaling_factor

    ### sanity check results
    out_encoded_df, _, _ = prepare_data_for_modeling(output_df, feature_columns, feature_encoding_map, target_column=outcome_key)
    new_cols = [col[:-len('_behavior_present')] if col.endswith('_behavior_present') else col for col in out_encoded_df.columns]
    out_encoded_df.columns = new_cols
    out_encoded_df[outcome_key] = cp.deepcopy(output_df[outcome_key])

    features = []
    presence_means = []
    n_not_dict = collections.OrderedDict([
            ('before', []),
            ('after', []),
             ])
    n_autism_dict = collections.OrderedDict([
            ('before', []),
            ('after', []),
            ('before_weighted', []),
            ('after_weighted', []),
             ])
    autism_frac_dict = collections.OrderedDict([
            ('before', []),
            ('after', []),
            ('before_weighted', []),
            ('after_weighted', []),
    ])
    for instruct in instructions:
        feature = instruct['feature']
        if module is not None:
            if 'instrument_'+str(module) not in feature: continue

        if feature not in encoded_df.columns:
            continue
        features.append(feature)
        presence_means.append(instruct['presence_means'])
        no_presence_df = encoded_df[encoded_df[feature]==0][[feature, outcome_key]]
        out_no_presence_df = out_encoded_df[out_encoded_df[feature]==0][[feature, outcome_key]]

        print 'for instructions ', instruct, ', no_presence_df: ', no_presence_df
        n_not = float(len(no_presence_df[no_presence_df[outcome_key]=='not'].index))
        n_autism = float(len(no_presence_df[no_presence_df[outcome_key]=='autism'].index))
        print 'n_not: ', n_not, ', n_autism: ', n_autism
        n_autism_weighted = autism_scale_factor*n_autism
        autism_frac = n_autism / (n_not + n_autism)
        autism_frac_weighted = n_autism_weighted / (n_autism_weighted + n_not)
        n_not_out = float(len(out_no_presence_df[out_no_presence_df[outcome_key]=='not'].index))
        n_autism_out = float(len(out_no_presence_df[out_no_presence_df[outcome_key]=='autism'].index))
        n_autism_out_weighted = autism_scale_factor*n_autism_out
        autism_out_frac = n_autism_out / (n_autism_out + n_not_out)
        autism_out_frac_weighted = n_autism_out_weighted / (n_autism_out_weighted + n_not_out)


        n_not_dict['before'].append(n_not)
        n_not_dict['after'].append(n_not_out)
        n_autism_dict['before'].append(n_autism)
        n_autism_dict['before_weighted'].append(n_autism_weighted)
        n_autism_dict['after'].append(n_autism_out)
        n_autism_dict['after_weighted'].append(n_autism_out_weighted)
        autism_frac_dict['before'].append(autism_frac)
        autism_frac_dict['before_weighted'].append(autism_frac_weighted)
        autism_frac_dict['after'].append(autism_out_frac)
        autism_frac_dict['after_weighted'].append(autism_out_frac_weighted)

        #print 'For feature ', feature, ', no presence case input was n_not: ', n_not, ', n_autism: ', n_autism, ', weighted n_autism: ', (n_autism*autism_prior_scaling_factor)
        #print 'And output case was n_not: ', n_not_out, ', autism: ', n_autism_out, ', scaled autism: ', (n_autism_out*autism_prior_scaling_factor)
    draw_sanity_overlays(n_not_dict, features, presence_means, suspicious_features, title='Number of not autism results', ylabel='Number of children when feature not present', ylims=None)
    draw_sanity_overlays(n_autism_dict, features, presence_means, suspicious_features, title='Number of autism results', ylabel='Number of children when feature not present', ylims=None)
    draw_sanity_overlays(autism_frac_dict, features, presence_means, suspicious_features, title='Autism frac results', ylabel='Autism fraction when feature not present', ylims=[0., 1.4], draw_comp_line=0.5)

def draw_sanity_overlays(results_dict, feature_columns, presence_means, suspicious_features, title, ylabel, ylims, draw_comp_line=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,8))
    plt.grid(True)
    colors = ['red', 'blue', 'black', 'purple']
    base_XVals = np.arange(len(feature_columns))+0.5
    plot_num=0
    n_plots = len(results_dict.keys())
    for (leg_label, yVals), color in zip(results_dict.iteritems(), colors):
        xWidths = 1. / (n_plots + 2.)
        these_xVals = np.arange(len(feature_columns))+(float(plot_num)*xWidths)
        #print 'do xVals: ', these_xVals, ', yVals: ', yVals, ', color: ', color, ', leg_label: ', leg_label
        plt.bar(these_xVals, yVals, xWidths, color=color, label=leg_label)
        plot_num+=1

    if ylims is None:
        cur_ylims = plt.gca().get_ylim()
        ylim_range = cur_ylims[1] - cur_ylims[0]
        ylims = [0, cur_ylims[0]+(ylim_range*1.2)]
    plt.gca().set_ylim(ylims)
    if draw_comp_line is not None:
        xlims = plt.gca().get_xlim()
        plt.plot(xlims, [draw_comp_line]*2, color='red', linestyle='--', linewidth=2)
    plt.legend(fontsize=24)
    plt.xticks(base_XVals, feature_columns, rotation=70, fontsize=18)

    autism_features = []
    not_features = []
    for feature, presence_type in zip(feature_columns, presence_means):
        if presence_type=='not':
            not_features.append(feature)
        elif presence_type=='autism':
            autism_features.append(feature)
        else:
            assert 0


    print 'suspicious_features: ', suspicious_features
    for xtick_label in plt.gca().get_xticklabels():
        if xtick_label.get_text() in not_features:
            xtick_label.set_color('red')
        if xtick_label.get_text() in suspicious_features:
            print 'set label ', xtick_label.get_text(), ' to bold'
            xtick_label.set_weight('bold')
    plt.ylabel(ylabel, fontsize=20)
    plt.title(title, fontsize=22)
    plt.show()


def reverse_one_hot_encoding(dataset, input_columns, fallback_value='Unknown non-ASD', override_column=None):
    ''' Converts many columns that encode a feature with 1/0 values into a single
    column that reports one of these features as being present (needed for machine learning target).
    In the case where more than one of the input columns has a non zero value the first one
    in the list that is non zero will have priority.
    
    input_columns is a list of columns '1' or '0' values to combine.
    fallback_value is what should be specified if none of the input columns have a '1'

    if override_column is specified and relevant values are present then those take precedent over anything
    else in this function

	An example of using this function is a situation where you have many variables encoding a condition
	as a 0 or 1 (for example, does child have OCD, does child have ADHD, does child have autism, ...), and you want to 
	combine into a single column that summarizes what the most important of those conditions is (if
	any are non-zero), where importance is defined by the order of the columns. It is suggested, however, that
	you make the input columns non-overlapping where possible (for example, ['Neurotypical', 'Any Delay not ASD not ADHD', 'ADD/ADHD/OCD not ASD'])
    '''

    def pick_combined_value_for_row(row):
        possible_unknown_values = ['', 'NaN', 'missing']
        combined_value = None
        if override_column is not None:
            if row[override_column] not in possible_unknown_values:
                return row[override_column]
        for possible_combined_value in input_columns:
            if row[possible_combined_value] in possible_unknown_values:
                continue
            if int(row[possible_combined_value])==1:
                combined_value=possible_combined_value
                break
        if combined_value is None:
            combined_value=fallback_value
        return combined_value
    columns_for_combination = input_columns
    if override_column is not None:
        columns_for_combination.append(override_column)
    combined_series = dataset[columns_for_combination].apply(pick_combined_value_for_row, axis=1)
    return combined_series


def get_desired_condition_fracs(app_frac_df, training_set_df, non_target_cols, unknown_non_target_col,
        assumed_unknown_non_target_fracs, target_frac=0.5, reg_param=0.5, target_key='ASD', debug_level=0):
    ''' 
	This function is needed to determine the various non-ASD condition breakdowns for purposes of
	balancing them before training

	Function to determine the desired fractions (or weights) of each type of diagnosis, given a desired population that we want
    to configure our algorithms for (specified by app_frac_df).

    The function takes into account the starting fractions of the conditions (specified in training_set_df) and a desired weighting for
    the target_condition (probably want 0.5 to fit this versus everything else). The function also allows for a sample of 'unknown'
    condition in the training set. When this is present it needs a guess for the breakdown of conditions within the unknown population,
    specified by assumed_unknown_non_target_fracs

    The function will do a ridge regression with regularization parameter=reg_param to pick a good set of target weights, apply some constraints,
    and then return a set of target weights that respect target_frac, and attempt to bring the fractions as close as possible to the desired
    app_frac_df without making any condition have weights that are too far out of line with their starting values (maximum weight increase of 5x by default)

    All dataframes are assumed to be binned by age of the child. Rows are the child age, columns are different conditions, and the values are the fractions
    of children of that age with a given condition

    detailed description of the inputs:
        app_frac_df: desired fractions of children with each condition, binned by child age (rows). Conditions are in the columns and entries are the fractions.
        training_set_df: starting fractions of the conditions in the training set. Organized as app_frac_df
        non_target_cols: a list that specifies which columns of the training_set_df refer to conditions that are spectators to the target condition and which 
            must have their fractions determiend
        unknown_non_target_col: Optional. If your training data contains a children of "unknown" conditions (must be sure they do not have target condition),
            specify that column here.
        assumed_unknown_non_target_fracs: For your children with unknown (but not target) condition, this is your guess of their condition breakdown, organized
            as a dataframe that is structured like app_frac_df
        target_frac: a fraction to represent what the weighting of the target condition should be (probably want 50% here?)
        reg_param: when running ridge regression, how strong should the regularization be. The larger, the less the training data fractions can be shifted.
        target_key: which condition will be considered the target (default is autism)
        debug_level: the larger this number, the more verbose the printing in this function
    '''

    def get_X_y_vals_for_diagnosis_frac_fit(app_frac_vals, unknown_frac_vals, initial_training_fracs, target_frac):
        ''' This reformats inputs into the matrices that need to be
        passed to the ridge regression, with proper sklearn format '''
        ####################
        ### Example X-matrix for 3 non-autism categories should look like this:
        ### 
        ###    1.    0.    0.    f1_unknown
        ###    0.    1.    0.    f2_unknown
        ###    0.    0.    1.    f3_unknown
        ###    1.    1.    1.       1.
        ##### The fX_unknowns are fractions of the total non-autistic sample in the desired output
        ##### (total including autism after autism frac is set to target_frac)
        ##
        ## initial_training_fracs are used as offsets to make it so that regularization will constrain to change
        ## results as little as possible from default fractions
        ### Example y_vals looks like this:
        ###    f1_App - f1_train_initial, f2_App-f2_train_initial, f3_App-f3_train_initial, (1-f_autism-sum(f trains))   (but vertical rather than horizontal)
        ####################
    
        y_vals = list(app_frac_vals-initial_training_fracs) + [1. - target_frac - np.sum(initial_training_fracs)]    ### represents the constraint that all non-target + target adds to a total fraction of 1.0
        assert abs(np.sum(y_vals) + (2.*np.sum(initial_training_fracs))  - 1.) < 0.00001
        y_arr = np.array(y_vals).reshape(len(app_frac_vals)+1, 1)
    
        my_X_list = []
        for irow, row in enumerate(range(len(app_frac_vals))):
            build_this_row = [0.]*(irow) + [1.] + [0.]*(len(app_frac_vals)-(irow+1))
            ### Also append the unknown part for this dimension
            build_this_row.append(unknown_frac_vals[irow])
            my_X_list.append(build_this_row)
        my_X_list.append([1.]*(len(app_frac_vals)+1))
        X_arr = np.array(my_X_list)
        return X_arr, y_arr

    def get_tot_frac_non_target(in_df, non_target_cols):
        ''' Helper function assumes that in_df has non_target_cols filled with 
        values that represent fractions of total that include target '''
        non_target_df = cp.deepcopy(in_df)
        non_target_df = non_target_df.drop(target_key, 1)
        tot_frac_non_target = cp.deepcopy(in_df[non_target_cols[0]])
        for col in non_target_cols[1:]:
            tot_frac_non_target += in_df[col]
        return tot_frac_non_target
    def enforce_constraints(fracs_dict, target_frac, initial_training_non_target_fracs_dict, max_allowed_weighting=5.):
        ### first get rid of negatives:
        for key in fracs_dict.keys():
            if fracs_dict[key] < 0.: fracs_dict[key] = 0.
        ### Now enforce maximum weighting 
        for key in fracs_dict.keys():
            if key==target_key: continue
            if fracs_dict[key] > max_allowed_weighting*initial_training_non_target_fracs_dict[key]:
                fracs_dict[key] = max_allowed_weighting*initial_training_non_target_fracs_dict[key]
        ### Now normalize so that total is 1.0
        expected_non_target_total = 1. - target_frac
        actual_non_target_total = np.sum([value for key, value in fracs_dict.iteritems() if key != target_key])
        correction_factor = expected_non_target_total / actual_non_target_total
        for key in fracs_dict.keys():
            if key == target_key: continue
            fracs_dict[key] *= correction_factor
        return fracs_dict

    age_groups = app_frac_df['age_category'].values
    desired_condition_fracs = {condition: [] for condition in non_target_cols+[unknown_non_target_col]}
    desired_condition_fracs[target_key] = []
    desired_condition_fracs['age_category'] = []
    for age_group in age_groups:

        ### Do linear regression of N+1 equations with N+1 unknowns where the N corresponds to the number of non-autism conditions
        ### And the +1 represents the unknown category
        app_non_target_app_frac_vals = np.array([app_frac_df[app_frac_df['age_category']==age_group][non_target_condition].values[0] for non_target_condition in non_target_cols])
        desired_non_target_fraction = 1. - target_frac
        desired_non_target_frac_vals = app_non_target_app_frac_vals * (desired_non_target_fraction / np.sum(app_non_target_app_frac_vals))
        if len(training_set_df[training_set_df['age_category']==age_group].index)==0:
            print 'No data available for ', age_group
            continue
        training_non_target_frac_vals = np.array([training_set_df[training_set_df['age_category']==age_group][non_target_condition].values[0] for non_target_condition in non_target_cols])
        unknown_frac_vals = [assumed_unknown_non_target_fracs[assumed_unknown_non_target_fracs['age_category']==age_group][non_target_condition].values[0] for non_target_condition in non_target_cols]
        X_arr, y_arr = get_X_y_vals_for_diagnosis_frac_fit(desired_non_target_frac_vals, unknown_frac_vals, initial_training_fracs=training_non_target_frac_vals, target_frac=target_frac)

        #### Now do ridge regression
        if debug_level>=2:
            print 'age_group: ', age_group, ', do fit on: '
            print 'X: ', X_arr
            print 'y: ', y_arr
        reg = linear_model.Ridge(alpha=reg_param, fit_intercept=False)
        fit_output = reg.fit(X_arr, y_arr)
        if debug_level>=2:
            print 'fit intercept: ', fit_output.intercept_
            print 'fit parameters: ', fit_output.coef_
            print 'initial training offsets were: ', training_non_target_frac_vals
            print '(Need to add those back in to get real app values)'

        ### Now undo the centering of the fit parameters around the initial values
        output_fracs_dict = {}
        for idx, (non_target_condition, frac) in enumerate(zip(non_target_cols+[unknown_non_target_col], fit_output.coef_[0])):
            if non_target_condition != unknown_non_target_col: frac += training_non_target_frac_vals[idx]
            output_fracs_dict[non_target_condition] = frac
        output_fracs_dict[target_key] = target_frac

        ### Now enforce reasonable constraints on output weighting
        if debug_level>=2:
            print 'Raw output_fracs before enforcement of constraints: ', output_fracs_dict
        initial_training_non_target_fracs_dict = {condition: frac for condition, frac in zip(non_target_cols, training_non_target_frac_vals)}
        initial_training_non_target_fracs_dict[unknown_non_target_col] = training_set_df[training_set_df['age_category']==age_group][unknown_non_target_col].values[0]
        output_fracs_dict = enforce_constraints(output_fracs_dict, target_frac, initial_training_non_target_fracs_dict)
        if debug_level>=2:
            print 'output_fracs_ after enforcement of constraints: ', output_fracs_dict
        assert abs(np.sum(output_fracs_dict.values()) - 1.) < 0.00001
        for condition, frac in output_fracs_dict.iteritems():
            desired_condition_fracs[condition].append(output_fracs_dict[condition])
        desired_condition_fracs['age_category'].append(age_group)
    desired_fracs_df = pd.DataFrame(desired_condition_fracs)
    if debug_level>=1:
        debug_cols = ['age_category', target_key]+non_target_cols 
        print 'Show results for: '
        print 'target column: ', target_key
        print 'input training fracs: ', training_set_df[debug_cols+[unknown_non_target_col]]
        print 'desired App Fracs: ', app_frac_df[debug_cols]
        print 'assumed unknown composition: ', assumed_unknown_non_target_fracs[['age_category']+non_target_cols]
        print 'And the results are: ', desired_fracs_df[debug_cols+[unknown_non_target_col]]
    return desired_fracs_df



    

