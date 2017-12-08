
# coding: utf-8

# # Create your own FLS in Python
# 
# ## Lab session 1
# 
# In this lab we provide you with the basics for creating your own Fuzzy Logic System (FLS). We will examine the theory and put it to practice at the same time.
# 
# ### Theory
# 
# Let us start with an overview of all the components of an FLS:
# 
# <img src="https://i.imgur.com/blP93rN.png"></img>
# 
# The <b>fuzzifier</b> needs:
# - input variables, output variables
# - membership functions for its variables
# 
# The <b>rules</b> include:
# - an antecedent containing a selected mf per variable (e.g. if x1 is low ...)
# - operators: and, or, not (in this lab we only consider one operator in a rule's antecedent)
# - a consequent containing a selected mf for the output variable (Mamdani) or a formula for the output variable (TSK)
# 
# The <b>inference mechanism</b> does:
# - uses the firing strength of a rule to calculate the result of the implication
# - (for which it uses an implication operator: min, prod)
# - aggregates the results
# 
# The aggregated result is then <b>defuzzified</b> and becomes a crisp value.
# 
# 
# ### Practice
# 
# Will make a start with implementing your own FLS. This can be of great help for your project, for which you will - hopefully - implement your own FLS too, and for your understanding of Fuzzy Logic.
# 
# We are going to create a Mamdani inference system that helps you decide how much money to put aside per month for buying a new laptop / TV / headphone:
# - input variables: income (0-1000), and the quality of your current device (0-10).
# - output variable: amount of money (1-500)
# 
# This tutorial is <b>by no means a perfect</b> implementation of an FLS, it is a very simple one and would have to be adapted for being usable for the FL projects.

# ## 1. Membership Functions
# 
# ### Theory
# 
# Membership functions are used to fuzzify crisp inputs, representing an item's membership to a class, with 0 meaning no membership and 1 meaning the item is a perfect prototype of a class.
# 
# During the Fuzzy Logic crash course we have seen 4 basic types of membership functions: triangular, trapezoidal, gaussian and generalized bell shaped membership functions. Those membership functions are represented in the image:
# 
# <img src="https://i.imgur.com/zwHXTsq.png"></img>
# 
# ### Practice
# 
# Implement the trapezoidal and triangular membership functions by completing the code below. The *calculate_membership()* function should return a scalar.
# 
# Test your implementation by submitting your answers via https://docs.google.com/forms/d/e/1FAIpQLScUoATQNzF30YSLxJUeoFHdJdtM4DK6bPA4oIpmczqTH0FDDg/viewform?usp=sf_link.

# In[2]:

from __future__ import division
import math
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt


# In[7]:

class TriangularMF:
    """Triangular fuzzy logic membership function class."""
    def __init__(self, name, start, top, end):
        self.name = name
        self.start = start
        self.top = top
        self.end = end

    def calculate_membership(self, x):
        if x<self.start or x>self.end:
            return 0
        if x>self.top:
            return (self.end-x)/(self.end-self.top)
        return (x-self.start)/(self.top-self.start)


class TrapezoidalMF:
    """Trapezoidal fuzzy logic membership function class."""
    def __init__(self, name, start, left_top, right_top, end):
        self.name = name
        self.start = start
        self.left_top = left_top
        self.right_top = right_top
        self.end = end

    def calculate_membership(self, x):
        if x<self.start or x>self.end:
            return 0
        if x>=self.left_top and x<=self.right_top:
            return 1
        if x>self.right_top:
            return (self.end-x)/(self.end-self.right_top)
        return (x-self.start)/(self.left_top-self.start)


# In[4]:

x_list = range(0,30)
trapezoid = TrapezoidalMF('I member', 7, 10, 18, 25)
y_list = [trapezoid.calculate_membership(x) for x in x_list]
plt.plot(x_list, y_list)
plt.show()


# In[5]:

# Test your implementation by running the following statements
# Enter your answers in the Google form to check them, round to two decimals

triangular_mf = TriangularMF("medium", 150, 250, 350)
print(triangular_mf.calculate_membership(100))
print(triangular_mf.calculate_membership(249))
print(triangular_mf.calculate_membership(300))

trapezoidal_mf = TrapezoidalMF("bad", 0, 0, 2, 4)
print(trapezoidal_mf.calculate_membership(1.2))
print(trapezoidal_mf.calculate_membership(2.3))
print(trapezoidal_mf.calculate_membership(3.9))


# ## 2. Inputs and output
# 
# ### Theory
# 
# The inputs and output of a FLS are represented through linguistic variables, which are variables whose values are words rather
# than numbers. A value of a linguistic variable is called a linguistic term.
# 
# An example of a linguistic variable is 'income' (a variable we're using in the system we're programming), with the linguistic terms 'low', 'medium', 'high'.
# 
# ### Practice
# 
# Now we are going to define input and output variables, which are a collection of multiple membership functions.
# 
# A variable's membership functions (self.mfs) should be a list of membership functions. Define the input variables *income* and *quality* and the output variable *money* with the name, range and membership functions represented in the image. Afterwards, check your answers through the google form.
# 
# (Note that you can use the trapezoidal membership function to represent a left or right shoulder membership function.)
# 
# <img src="https://i.imgur.com/7iIyCco.png"></img>

# In[11]:

class Variable(object):
    """General class for variables in an FLS."""
    def __init__(self, name, range, mfs):
        self.name = name
        self.range = range
        self.mfs = mfs

    def calculate_memberships(self, x):
        """Test function to check whether
        you put together the right mfs in your variables."""
        return {
            mf.name : mf.calculate_membership(x)
            for mf in self.mfs
        }

    def get_mf_by_name(self, name):
        for mf in self.mfs:
            if mf.name == name:
                return mf

class Input(Variable):
    """Class for input variables, inherits 
    variables and functions from superclass Variable."""
    def __init__(self, name, range, mfs):
        super(Input, self).__init__(name, range, mfs)
        self.type = "input"

class Output(Variable):
    """Class for output variables, inherits 
    variables and functions from superclass Variable."""
    def __init__(self, name, range, mfs):
        super(Output, self).__init__(name, range, mfs)
        self.type = "output"


# In[12]:

# Input variable for your income
# Your code here
mfs_income = [TrapezoidalMF('Low', -200, 0, 200, 400),
              TriangularMF('Medium', 200, 500, 800),
              TrapezoidalMF('High', 600, 800, 1000, 1200)]
income = Input("income", (0, 1000), mfs_income)

# Input variable for the quality
# Your code here
mfs_quality = [TrapezoidalMF('Bad', -2, 0, 2, 4),
              TriangularMF('Okay', 2, 5, 8),
              TrapezoidalMF('Amazing', 6, 8, 10, 12)]
quality = Input("quality", (0, 10), mfs_quality)

# Output variable for the amount of money
# Your code here
mfs_money = [TrapezoidalMF('Low', -150, 0, 100, 250),
            TriangularMF('Medium', 150, 250, 350),
            TrapezoidalMF('High', 250, 400, 500, 650)]
money = Output("money", (0, 500), mfs_money)

inputs = [income, quality]
output = money


# In[14]:

# Test your implementation by running the following statements
# Enter your answers in the Google form to check them, round to two decimals

print(income.calculate_memberships(489))
print(quality.calculate_memberships(6))
print(output.calculate_memberships(222))


# ## 3. Fuzzy Rules
# 
# ### Theory
# 
# A fuzzy IF-THEN rule is composed of an antecedent and a consequent, like the implications you have seen with propositional and first order logic.
# - In the antecedent conditions for input variables are connected with an operator (AND, OR, NOT), e.g. "IF x1 is mf1 AND x2 is mf3", or "IF x1 is mf1 OR x2 is mf2".
# - The AND operator represents taking the intersection of fuzzy sets, which can be accomplished by choosing a T-Norm operation, such as minimum.
# - The OR operator represents taking the union of fuzzy sets, which can be accomplished by choosing a T-Conorm operation, such as maximum.
# - The NOT operator, the complement, is calculated by 1 minus a membership value. For example: NOT x1 is mf1, would be 1 - (membership of x1 to mf1).
# - By the use of the AND, NOT and OR we combine the different parts of the antecedent into a single number, which is the firing strength of the antecedent.
# - The consequent represents an action that we undertake if the rule fires.
# 
# ### Practice
# 
# We are going to add some simple rules for our FLS: complete rules that do not have mixed operators. Here we represent a rule through 3 variables:
# - <b>Antecedent</b>, represented as a list of names of membership functions. The index of the name corresponds to the variable it belongs to, for example: ["medium", "low"], where "medium" belongs to the first variable in *inputs* and "low" corresponds to the second variable in *inputs*.
# - <b>Operator</b>: "and" or "or", let's choose "and".
# - <b>Consequent</b>: a string corresponding one of the membership functions of your output variable, for example "high".
# 
# These three variables would then compose the rule "IF income is medium AND quality is low THEN money is high."
# 
# Complete the *calculate_firing_strength()*, that should function and check your answers by running the test statements.

# In[21]:

class Rule:
    """Fuzzy rule class, initialized with an antecedent (list of strings),
    operator (string) and consequent (string)."""
    def __init__(self, n, antecedent, operator, consequent):
        self.number = n
        self.antecedent = antecedent
        self.operator = operator
        self.consequent = consequent
        self.firing_strength = 0
        if self.operator == 'and':
            self.operation = lambda x: min(x)
        elif self.operator == 'or':
            self.operation = lambda x: max(x)

    def calculate_firing_strength(self, datapoint, inputs):
        return self.operation([inputs[i].calculate_memberships(datapoint[i])[self.antecedent[i]] for i in range(len(inputs))])


# In[22]:

# Test your implementation by checking the following statements
# Enter your answers in the Google form to check them, round to two decimals

rule1 = Rule(1, ["Low", "Amazing"], "and", "Low")
print(rule1.calculate_firing_strength([200, 6.5], inputs))
print(rule1.calculate_firing_strength([0, 10], inputs))

rule2 = Rule(2, ["High", "Bad"], "and", "High")
print(rule2.calculate_firing_strength([100, 8], inputs))
print(rule2.calculate_firing_strength([700, 3], inputs))


# ## 4. Fuzzy Rulebase
# 
# 
# ### Theory
# 
# A rulebase is simply a collection of all rules of the system!
# 
# ### Practice
# 
# Our fuzzy rulebase is a collection of all rules. Create the following rules and initalize the fuzzy rulebase:
# - IF income is low AND quality is amazing THEN money is low
# - IF income is medium AND quality is amazing THEN money is low
# - IF income is high AND quality is amazing THEN money is low
# - IF income is low AND quality is okay THEN money is low
# - IF income is medium AND quality is okay THEN money is medium
# - IF income is high AND quality is okay THEN money is medium
# - IF income is low AND quality is bad THEN money is low
# - IF income is medium AND quality is bad THEN money is medium
# - IF income is high AND quality is bad THEN money is high
# 
# Implement the *calculate_firing_strengths()* function that collects the highest firing strength found per membership function of the output variable in a dictionary or Counter object.
# For example, if the firing strengths for the rules listed above are 0, 0, 0, 0.5, 0.25, 0, 0, 0, 0 the result would look like this: *{"low":0.5, "medium":0.25, "high"0}*.
# 
# Check the correctness of your function with the testing statements.

# In[23]:

from collections import Counter

class Rulebase:
    """The fuzzy rulebase collects all rules for the FLS, can
    calculate the firing strengths of its rules."""
    def __init__(self, rules):
        self.rules = rules

    def calculate_firing_strengths(self, datapoint, inputs):
        result = Counter()
        for i, rule in enumerate(self.rules):
            fs = rule.calculate_firing_strength(datapoint, inputs)
            consequent = rule.consequent
            if not result.has_key(consequent) or fs > result[consequent]:
                result[consequent] = fs
        return result


# In[24]:

# Add the rules listed in the question description
# Your code here
rules = [Rule(1, ['Low', 'Amazing'], 'and', 'Low'),
        Rule(2, ['Medium', 'Amazing'], 'and', 'Low'),
        Rule(3, ['High', 'Amazing'], 'and', 'Low'),
        Rule(4, ['Low', 'Okay'], 'and', 'Low'),
        Rule(5, ['Medium', 'Okay'], 'and', 'Medium'),
        Rule(6, ['High', 'Okay'], 'and', 'Medium'),
        Rule(7, ['Low', 'Bad'], 'and', 'Low'),
        Rule(8, ['Medium', 'Bad'], 'and', 'Medium'),
        Rule(9, ['High', 'Bad'], 'and', 'High')]

rulebase = Rulebase(rules)


# In[25]:

# Test your implementation of calculate_firing_strengths()
# Enter your answers in the Google form to check them, round to two decimals

datapoint = [500, 3]
print(rulebase.calculate_firing_strengths(datapoint, inputs))

datapoint = [234, 7.5]
print(rulebase.calculate_firing_strengths(datapoint, inputs))


# ## 5. Inference (aggregation and defuzzification)
# 
# ### Theory
# 
# In the Fuzzy Inference all parts of the fuzzy system come together: we are mapping an input to an output in the following way:
# 1. Fuzzify the input.
# 2. Calculate the firing strengths for the rules.
# 3. Use the firing strength to determine the contribution of the consequent.
# 4. Aggregate / collect all consequents.
# 5. Defuzzify
# 
# As already mentioned, there is Mamdani type inference and Takagi-Sugeno-Kang (TSK) type inference:
# - With Mamdani type inference we represent the consequents of fuzzy rules as fuzzy sets (using membership functions). We use a rule's firing strength to adapt the height of the membership function in the consequent, using the implication operator (minimum or product). The consequents are then aggregated into one area (taking the maximum of all consequents for the entire input range), on which we apply a defuzzification method, such as 'largest of max', 'smallest of max' or 'centroid'.
# - With TSK type inference we represent the consequents as a function of the input variables, or a constant. To combine the consequents into one output number we calculate a weighted average, where the weights are the rules' firing strengths.
# 
# <img src="https://i.imgur.com/q5lzbsZ.png"></img>
# <img src="https://i.imgur.com/Yl20dJL.png"></img>
# 
# In the following image multiple defuzzification methods are visualized:
# <img src="http://access.feld.cvut.cz/storage/201208252026_obr-15.png"></img>
# 
# ### Practice
# 
# We will finalize our system using Mamdani type inference by performing the following three steps:
# 1. Gathering the largest firing strength per membership function of the output variable (implemented in your rulebase in Step 3)
# 2. Discretizing the range of your output variable and applying the aggregation method (<b>max</b>): for every bin you find the maximum fuzzy membership value. Notice that the membership functions of the output variable are `cut off' according to the firing strengths, with the implication method (<b>min</b>).
# To accomplish this we perform two steps:
#     - First we find where the aggregated area starts and ends on the x-axis
#     - Second we discretize the area between start and end into 201 points (thus representing the area in 200 bins)
# 3. Applying two defuzzification methods: implement smallest of max (<b>som</b>) and largest of max (<b>lom</b>).

# In[26]:

class Reasoner:
    def __init__(self, rulebase, inputs, output, n_points, defuzzification):
        self.rulebase = rulebase
        self.inputs = inputs
        self.output = output
        self.discretize = n_points
        self.defuzzification = defuzzification

    def inference(self, datapoint):
        # 1. Calculate the highest firing strength found in the rules per 
        # membership function of the output variable
        # looks like: {"low":0.5, "medium":0.25, "high":0}
        
        # Your code here
        #raise NotImplementedError
        firing_strengths = self.rulebase.calculate_firing_strengths(datapoint, self.inputs)
        print(firing_strengths)

        # 2. Aggregate and discretize
        # looks like: [(0.0, 1), (1.2437810945273631, 1), (2.4875621890547261, 1), (3.7313432835820892, 1), ...]
        input_value_pairs = self.aggregate(firing_strengths)

        # 3. Defuzzify
        # looks like a scalar
        crisp_output = self.defuzzify(input_value_pairs)
        return crisp_output

    def aggregate(self, firing_strengths):
        
        # First find where the aggregated area starts and ends
        # Your code here
        # raise NotImplementedError
        area = np.linspace(min([mf.start for mf in self.output.mfs]), max([mf.end for mf in self.output.mfs]), self.discretize)
        
        
        # Second discretize this area and aggragate
        # Your code here
        # raise NotImplementedError
        pts = []
#        for pt in area:
#            pts.append((pt, max([min(firing_strengths[mf.name], mf.calculate_membership(pt)) for mf in self.output.mfs])))
        

        return pts

    def defuzzify(self, input_value_pairs):
        # Your code here
        # raise NotImplementedError
        if self.defuzzification == 'som':
        return crisp_value


# In[27]:

# Test your implementation of the fuzzy inference
# Enter your answers in the Google form to check them, round to two decimals

thinker = Reasoner(rulebase, inputs, output, 201, "som")
datapoint = [100, 1]
print(round(thinker.inference(datapoint)))

thinker = Reasoner(rulebase, inputs, output, 101, "lom")
datapoint = [550, 4.5]
print(round(thinker.inference(datapoint)))

thinker = Reasoner(rulebase, inputs, output, 201, "som")
datapoint = [900, 6.5]
print(round(thinker.inference(datapoint)))

thinker = Reasoner(rulebase, inputs, output, 201, "lom")
datapoint = [100, 1]
print(round(thinker.inference(datapoint)))

thinker = Reasoner(rulebase, inputs, output, 101, "lom")
datapoint = [550, 4.5]
print(round(thinker.inference(datapoint)))

thinker = Reasoner(rulebase, inputs, output, 201, "lom")
datapoint = [900, 6.5]
print(round(thinker.inference(datapoint)))


# In[ ]:



