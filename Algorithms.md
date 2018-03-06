---


---

<h1 id="lets-get-esoteric">Let’s Get Esoteric:</h1>
<p>Math is the language of our universe, however according to physicists Joscha Bach (<a href="http://bach.ai">http://bach.ai</a>) , the universe is “not mathematical, but computational.”</p>
<p>Why? Because Bach argues matter always computes…<br>
Argument aside computations need 2 things:</p>
<ol>
<li>Data to compute</li>
<li>An algorithm to follow</li>
</ol>
<p>The linear regression often gets it’s 15 minutes of fame in the introduction period of a class. It’s cool - until we learn how many more algorithms there are!</p>
<p>I’m taking the time to appreciate this algorithm because it’s the foundation. Herein lies the secret sauce to pretty much all other algorithms - we just usually don’t dive deeply enough into it to notice.</p>
<p>Conceptually, linear regression is the logical process of using an equation to express the relationship between variables.</p>
<p>So, to recap, math is the language of the universe and algorithms are like our sentence structures.</p>
<p><img src="https://lh3.googleusercontent.com/-5slh4z7BgQnYMLoHhRBFVXJhnLAMOHoYu5vp8rvB6KzbbkkcsPtAAuPgL9-kKuTv_a7zfUgaF07" alt="enter image description here"></p>
<h1 id="the-lineage-of-linear-regression">The Lineage of Linear Regression</h1>
<p><em>Math brought us to the super field of <strong>Mathematical Decision Making</strong></em></p>
<blockquote>
<p><em>The Universe beget Mathematics, Mathematics beget Mathematical Decision Making, Mathematical Decision Making beget AI, AI beget Machine Learning, Machine Learning beget Supervised Learning, Supervised Learning beget Regression.</em></p>
</blockquote>
<ul>
<li>It basically encompasses the entirety of mathematics - even down to<br>
operation systems and econometrics.</li>
</ul>
<p>Within this field we find Artificial Intelligence… and within AI we find Machine Learning</p>
<p><img src="https://lh3.googleusercontent.com/VJ1dwZ1007qwYnkE7n3s0e-xEufiRGVccoMrPpLfP08S1oxVMKXPRSKce7FPjL7_m5vljzIjtqTu" alt="enter image description here"></p>
<blockquote>
<p>(<a href="https://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/">https://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/</a><br>
<a href="http://ocdevel.com/podcasts/machine-learning/4">http://ocdevel.com/podcasts/machine-learning/4</a>)</p>
</blockquote>
<h2 id="machine-learning">Machine Learning</h2>
<p>ML breaks down into 3 subcategories</p>
<h3 id="the-3-types-of-machine-learning">The 3 Types of Machine Learning:</h3>
<ol>
<li>
<p><strong>Supervised Learning:</strong> <em>You</em> train and evaluate the data.</p>
<ul>
<li>Linear regression</li>
<li>Logistic regression</li>
</ul>
</li>
<li>
<p><strong>Unsupervised Learning:</strong> The <em>algorithm</em> segments/evaluates the data.</p>
</li>
</ol>
<ul>
<li>The machine tries to find out and segment what is common or uncommon amongst the data
<ul>
<li>Market segmentation</li>
</ul>
</li>
</ul>
<ol start="3">
<li><strong>Reinforcement Learning:</strong> “The real AI in Machine Learning”</li>
</ol>
<ul>
<li>The machine either achieves or does not achieve a goal and receives a reward or punishment, respectively. Based on the outcome the machine learns and tries again.</li>
</ul>
<p>As we move down the chain from supervised learning towards reinforcement learning there, is a substantial leap in complexity, skills and experience levels needed to work in this space.</p>
<p>So, our starting point is rooted in <em>supervised learning</em>. Within this type of learning we can further categorise algorithms into <em>regression</em> or <em>classification</em> problems.</p>
<ul>
<li><strong>Classifiers</strong> tell you the <em>class</em> of a thing</li>
<li><strong>Regressors</strong> tell you the  <em>influence</em> of a thing</li>
</ul>
<p>…We’ll dive deeper into this <em>“influence”</em> and what that means…</p>
<hr>
<h1 id="algorithms">Algorithms</h1>
<blockquote>
<p><strong>Algorithm</strong>: A process or set of rules to be followed in calculations or<br>
other problem-solving operations.</p>
</blockquote>
<p>The most bare bones process every ML algorithm mimics looks something like this:</p>
<p><img src="https://lh3.googleusercontent.com/GuMKEzAYuZijaHfnf-Z8mTZoAPe7DST5GHt9lIIDk7oN1qc403RoDyrNHas3wJ-ZN1O5fTqUlZjF" alt="enter image description here"></p>
<p><strong>Input</strong> = predictor variables (X)<br>
<strong>Transformation</strong> = an algorithm<br>
<strong>Output</strong> = Prediction for the target variable (Y)</p>
<p><em>An algorithm has one goal - to learn weights (parameters).</em></p>
<p>These weights represent the influence - or relationship - a coefficient has to the target variable.</p>
<p><strong>Example</strong><br>
<img src="https://lh3.googleusercontent.com/aqLIh83de7S2dUC_f7bLgBQ8bXi0oQa3pYN02RiSzrYH78Yxn9gG4rB5wIJ4A8aiLnFjWcaYQ-2X" alt="enter image description here"></p>
<hr>
<h1 id="the-basic-process-of-ml">The Basic Process of ML</h1>
<p><em>There are <em><strong>3</strong></em> key steps that take place in <strong>all</strong> ML problems:</em></p>
<ol>
<li>
<p><strong>Predict</strong> (Hypothesis, Estimate)</p>
</li>
<li>
<p><strong>Error</strong> (Loss)</p>
</li>
<li>
<p><strong>Learn</strong> (Train, Fit)</p>
</li>
</ol>
<h2 id="prediction">Prediction</h2>
<p>We make predictions through a <strong>hypothesis function</strong>. Every machine learning algorithm has some kind of hypothesis function.</p>
<p>The hypothesis function is important because it acts as a guide - within the hypothesis function is a set of parameters our model is trying to determine. Those parameters help us make predictions.</p>
<p><strong>Therefore:</strong><br>
The reason we train data is to derive a <strong>hypothesis function</strong> we can use to predict output.</p>
<hr>
<p><em>Before we look at the hypothesis function, let’s look at something more familiar…</em></p>
<hr>
<h3 id="algebra">Algebra</h3>
<p>Have you ever seen this formula?</p>
<blockquote>
<p>Y = m(x) + B</p>
<p>Where:</p>
<ul>
<li>Y is the dependent variable (what we want to find)</li>
<li>B is the Y intercept</li>
<li>m(x) is the slope</li>
</ul>
</blockquote>
<p>We usually come across this formula - called point-slope form -  in algebra when plotting graphs.</p>
<p>Turns out, this is the same formula used for the linear regression model.</p>
<h3 id="linear-regression-model">Linear Regression Model</h3>
<blockquote></blockquote>
<p><img src="https://lh3.googleusercontent.com/EmOHLSessf3ZclOt81Tlxaqgpd08FdQG2O1HUkrqmV8JNYOTaKXkTVDhUgoeDUsnohy6dVQdJJcF" alt="enter image description here"></p>
<blockquote>
<p>Where:</p>
<ul>
<li>Y is the dependent variable</li>
<li>B0 (B sub zero) is the Y intercept</li>
<li>B1 is the slope</li>
<li>X1 is the predictor variable</li>
</ul>
</blockquote>
<p><strong>B0</strong> and <strong>B1</strong> are <strong>parameters</strong>.</p>
<p><strong>Parameters</strong> are conditions that must be met to fulfill an operation. Remember, the goal of training data is to determine these parameters.</p>
<p><em>So, this linear regression equation sounds kind of like the hypothesis function?</em></p>
<ul>
<li>They both take parameters</li>
<li>The parameters will help us make predictions</li>
</ul>
<h3 id="the-hypothesis-function">The Hypothesis Function</h3>
<p>Again, the reason we train data is to derive a <strong>hypothesis function</strong> we can use to predict output.</p>
<p>The hypothesis function sounds like another math equation we need to memorise… but it is actually just another way of writing out the linear regression model, take a look:</p>
<blockquote>
<p><img src="https://lh3.googleusercontent.com/55de0gkhNmOWjZGjLIgD_Cl3435HinDqn8peprot5WlgZZY6Ois7wJ8Tti6QZeyV077RXX0ALnqW" alt="enter image descriptionhere"><br>
Where:</p>
<ul>
<li>h of theta (x) = Y</li>
<li>theta0 (theta sub zero) is the Y intercept</li>
<li>theta1 is the slope coefficient</li>
<li>x is the slope</li>
</ul>
</blockquote>
<p>The goal is to predict theta.</p>
<p><em>Theta is our parameter in this context.</em></p>
<p>Once these parameters are determined we can plug in some X values and calculate a prediction for Y.</p>
<hr>
<p><strong>Recap:</strong></p>
<ol>
<li>So we know that we need parameters, and that they are derived from the hypothesis function.</li>
<li>The hypothesis function helps us make predictions.</li>
</ol>
<hr>
<p><em><strong>But how do we get these parameters - are they just randomly determined?</strong></em></p>
<hr>
<h2 id="error">Error</h2>
<h3 id="the-cost-function">The Cost Function</h3>
<p>Also known as <strong>error</strong>.</p>
<p><em>Error is inherently apart of our linear regression equation.</em></p>
<blockquote>
<p><img src="https://lh3.googleusercontent.com/w150AlY_m2cDFT5fqzBNvJCdzD3VfNeIOV--LkHLOgOAqbMQzOL0h6gqO60FEEEkozmx_0WKiRuI" alt="enter image descriptionhere"></p>
<p>This is the same equation we saw above but with some more X variables<br>
and an epsilon.</p>
</blockquote>
<p><em>The epsilon is our error term.</em></p>
<p><strong>Why is error inherent?</strong><br>
Whenever we make a prediction we will most likely be slightly off from the truth. So, we take our prediction and find our error term with the following equation:</p>
<blockquote>
<p><img src="https://lh3.googleusercontent.com/nA_e2i00WE-r1X3kjjMMacWvRBHbvMHiXYpgNePWkdstdhjiIKBrOoYi77c-JyFqosd8SwCpscrW" alt="enter image descriptionhere"></p>
<p>This is also referred to as the <strong>residual</strong></p>
</blockquote>
<p><strong>Therefore:</strong><br>
Parameters are determined by the cost function, where error is minimised.</p>
<hr>
<p><strong>Recap:</strong></p>
<ol>
<li>We start with some data and our algorithm makes an initial random guess for each data point.</li>
<li>Then it looks at how wrong it was on each guess, iterates back our everything again in search of a set of parameters that are minimised (with the least amount of error).</li>
<li>Theses parameters are used in our hypothesis function to make predictions!</li>
</ol>
<h2 id="learn">Learn</h2>
<p>Our algorithm iterates over each data point and records the <strong>error</strong></p>
<ul>
<li>residuals = (actual - predicted Y values)</li>
</ul>
<p>Then it takes the sum of squares of that error and  makes a path through the data points where error is minimised across all data.</p>
<p>This is called:</p>
<blockquote>
<p><strong>The Line of Best Fit</strong>.</p>
<p><img src="https://lh3.googleusercontent.com/atO7ZSjNrgPnfjfH3_mSvl0DKvEbukhrmWXcLWDl3TZZUFWaOZbHlSoi9UKq-te3RE_h5I0-5PUb" alt="enter image descriptionhere"></p>
</blockquote>
<p><em>AKA: Our predictions.</em></p>
<hr>
<h2 id="important-notes">Important Notes</h2>
<p>Every machine learning algorithm has the same process we just went over.</p>
<ul>
<li>
<p>Predict</p>
</li>
<li>
<p>Error</p>
</li>
<li>
<p>Learn</p>
</li>
</ul>
<p>The Cost Function is often called the Optimization Function, and it uses something called Gradient Descent to help minimise error.</p>
<p><em>We won’t dive into Gradient Descent or other Optimization Functions today. However, these come into play with <em><strong>Neural Nets</strong></em> and <em><strong>AI</strong></em> algorithms!</em></p>
<hr>
<h1 id="building-a-linear-regressor-from-scratch">Building A Linear Regressor From Scratch</h1>
<p>The best way to really understand what’s going on in an algorithm is to use it!</p>
<p>Since linear regression can be done in just a few lines of code… this is a perfect opportunity to open the hood and see how it works.</p>
<h3 id="requirements-this-tutorial-assumes...">Requirements: This Tutorial Assumes…</h3>
<ol>
<li>You have <em><strong>Python 2.7 or above</strong></em> installed</li>
<li>You understand how to load data in R or Python via your file’s <em><strong>path</strong></em></li>
<li>You know how to install <em><strong>Python libraries</strong></em></li>
<li>The libraries I’ll be using:
<ul>
<li>Pandas - for dataframes</li>
<li>Matplotlib - for plotting our data and regression line</li>
<li>NumPy - (“Numeric Python”) - to do math</li>
<li>Sklearn - (“Scientific Python”) - for the linear regression algorithm</li>
</ul>
</li>
</ol>
<hr>
<h3 id="the-code">The Code:</h3>
<p><strong>Step 1:</strong>  Import Libraries:</p>
<pre><code> import pandas as pd 
 import matplotlib.pyplot as plt 
 import numpy as np
 %matplotlib inline 
 from sklearn import linear_model
</code></pre>
<p><strong>Step 2:</strong> Download Data:</p>
<ol>
<li>
<p>Download the file “BikesPerDay.csv” Included in this GitHub repo.</p>
</li>
<li>
<p>Save it somewhere on your computer (I saved it to my Desktop).</p>
</li>
<li>
<p>Set your path and read in the file.</p>
<pre><code>  path = "/Users/jessicaprim/Desktop/BikesPerDay.csv"
  bpd = pd.read_csv(path)
</code></pre>
</li>
</ol>
<p><strong>Step 3:</strong> Explore:</p>
<pre><code>bpd.head()
</code></pre>
<p><img src="https://lh3.googleusercontent.com/zDJCo_jXJHVvIrgTXyFkTWjtpgn4vpuryEqMT2yWBOVGKGQx5Nae0VJQlvXACEeXqeZ6ge4Ktzps" alt="enter image description here"></p>
<p>It’s good practice to keep a copy of your original dataset before you begin any transformations. This is mostly for insurance purposes… and you may need to make more copies down the road.</p>
<p>I will make a new data frame called ‘bike_data’ and copy the original data.</p>
<pre><code># Copy our dataset so we can subset 
# some columns while keeping original data

bike_data = bpd.copy()
</code></pre>
<hr>
<p>This next line might look a bit confusing. I am using a function called <strong>“loc”</strong>.</p>
<p>This is one way we can select specific columns from a data frame by label, or by index. Here, I am selecting columns by label.</p>
<p>I set the data frame equal to itself - so the transformation is immutable (permanent, unchanging) in bike_data.</p>
<pre><code># Indexing columns by label

bike_data = bike_data.loc[:, ['cnt','temp']]
</code></pre>
<p>I set the data frame equal to itself - so the transformation is immutable (permanent, unchanging) in bike_data.</p>
<hr>
<p>Check the data…</p>
<pre><code>bike_data.head()
</code></pre>
<p><img src="https://lh3.googleusercontent.com/I6VTDhK55xxWY10xxkNxspSCeEI41tnxZQQevVSe4WiNGyLq_UU772ifjzNvMygryMEoZLWUYljC" alt="enter image description here"></p>
<p>I chose to look at count and temperature. My goal is to see how well temperature can predict the quantity of bikes use per day.</p>
<p><strong>Step 4:</strong> Visualize Data:</p>
<pre><code># Visualization:

bike_data.plot(kind = "scatter",
          x = "temp",
          y = "cnt",
          color = "black")
</code></pre>
<p><img src="https://lh3.googleusercontent.com/MlxE4xZcXc5omeo4olQUOFSLx6ycPRU8HAVHnkyXPr4EhjHVES03nCvwlZPtmnr0Yf35ucY9IYrv" alt="enter image description here"></p>
<p>After plotting the temperature against count you might notice a mild positive relationship between the two. A warmer day might indicate a higher bike count.</p>
<p>I say “mild” and “might” because the spread of the data is pretty wide and looks a bit sporadic - so the relationship is fuzzy at best.</p>
<p><strong>Step 5:</strong> The Model:</p>
<pre><code># Initialize Model
regression = linear_model.LinearRegression()

#Fitting the model
regression.fit(X = pd.DataFrame(bike_data["temp"]), 
			    y = bike_data["cnt"])
</code></pre>
<ul>
<li>
<p>Line 1:<br>
<em>regression = linear_model.LinearRegression()</em></p>
<ul>
<li>We instantiated (created) an  object called ‘regression’ that uses the linear regression algorithm (LinearRegression() ).</li>
<li>LinearRegression() is a function.</li>
</ul>
</li>
<li>
<p>Line 2:<br>
<em>regression.fit()</em></p>
<ul>
<li>This function take 2 arguments, X and y. That means we have to enter values for these arguments - otherwise the algorithm won’t work. This is how the algorithm interprets our data.</li>
</ul>
<p><em>X = pd.DataFrame(bike_data[“temp”])</em></p>
<ul>
<li>This line feeds a value to the first argument, X.</li>
<li>X represents our predictor variable, “temp”.</li>
<li>The value we gave X  is read in as a data frame.</li>
</ul>
<p><em>y = bike_data[“cnt”])</em></p>
<ul>
<li>All we are doing here is referring to the column “cnt”, our target variable.</li>
</ul>
</li>
</ul>
<p>Now let’s look at the coefficients produced as a result of the regression!</p>
<pre><code># Y-Intercept
print(regression.intercept_)

#Slope
print(regression.coef_)
</code></pre>
<p><img src="https://lh3.googleusercontent.com/_YDOclWoZjXHfKM_ZtXktyNZrirqkcCJ8gMYmA0Jhz2potnVBhGAToF-0ZMaQ2CeY3s90RdRWyI7" alt="enter image description here"></p>
<p><strong>Step 6:</strong> How Well Did We Do?</p>
<p><em>We will look at R^2, (Residuals squared).</em></p>
<p><strong>R^2</strong> tells us how much the target variable can be explained by the predictor variables.</p>
<p>Check out the score (how well temperature predicts count)</p>
<pre><code>regression.score(X = pd.DataFrame(bike_data["temp"]),
 y = bike_data["cnt"])
</code></pre>
<p><img src="https://lh3.googleusercontent.com/wV_NzizHuqjqsaevwiv7GCaqmDpQf-G1VqVLbT3Dl6tu-xWPNBITrnqlpD8wEQQZ1Gx0pzwM-eeI" alt="enter image description here"></p>
<ul>
<li><strong>Note how similar the above code looks to the line we wrote for regression.</strong></li>
<li>Also note how bad our score is…</li>
</ul>
<p><strong>Step 7:</strong> Plot the Line of Best Fit:</p>
<pre><code>   bike_data.plot(kind = "scatter",
              x = "temp",
              y = "cnt",
              color = "black")

  #Regression Line
    plt.plot(bike_data["temp"],
            predictions,
            color = "red")
</code></pre>
<p><img src="https://lh3.googleusercontent.com/42yq82p0Gf-R3KyijSRA9gMCNQGcAALp8lU6hhe7gLUkQmfCnxYsSPUhdUKUCALUBX0TlrPANn-8" alt="enter image description here"></p>
<p><em>So, is Temperature a good predictor of how many bikes will be sold in a day?</em></p>
<ul>
<li>According to our low test score - probably not.</li>
</ul>
<p>This is where we add in some more features and run <strong>multivariate regression</strong>.</p>

