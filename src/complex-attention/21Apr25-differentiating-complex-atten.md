---
header-includes:
  - \usepackage{amsmath}  # Essential for advanced math formatting
  - \usepackage{amsfonts}  # For additional math fonts like blackboard bold
  - \usepackage{amssymb}   # For additional math symbols
geometry: margin=0.75in
papersize: a4paper
fontsize: 11pt
wrap: none
mainfont: "Liberation Serif"
sansfont: "Liberation Sans"
monofont: "Hack"
---

# Differentiating Complex-Valued Attention
This article is a brief, semi-formal walkthrough of the gradients involved in complex-valued attention. Complex-Valued Neural Networks (CVNNs) are a as-of-yet not fully explored region of deep learning owing to very niche potential applications, and the fact that most of the time their irrelevant for practical use-cases. Of most of the research I've found so far into CVNNs, very few networks exist that work purely in the complex domain, and even fewer - if any at all - have ever been deployed for a pragmatic business use-case. Most of the research applies complex-valued components as a supplement to real-valued components, and a few of these models have reached production. Almost all of the research I've seen involves either the FFT or a wavelet transformation.

A few reasons why CVNNs are not as ubiquitous as their real-valued counterparts are that their operations and workings are notoriously difficult to interpret, moreso than their real cousins. Complex gradients also depend on rather strict conditions toward the property of their parent functions being _holomorphic_, which even though grants a suite of powerful theoretical and analytical tools (contour integration, residue calculus, conformal mappings.), for a variety of pragmatic uses, is irrelevant. Lastly, complex analysis itself is (unfortunately) not a ubiquitously used area of mathematics, and so its extensions are understandably less pervasive.

A few reasons why CVNNs are worthy of further practical research - not into their development or application, but into understanding how the aforementioned limitations apply to potential use-cases - are that CVNNs encode a great deal of information compactly, moreso than real-valued neural nets can, owing to the properties of complex numbers themselves. Complex numbers - and therefore CVNNs - are extremely useful in manipulating periodic and/or rotating types of signals or signals that inherently depend on _phase_; they're also very powerful in manipulating image data at a fraction of the resources required by their real cousins. Their expressivity - complex numbers and CVNNs - also allows for an incredible depth of analyses to be carried out on single and multiple signals (time series); for example, techniques like pole-zero ARMA analysis, wavelet deconstruction, coherence analysis, and even complex-valued PCA/Singular Spectrum or bispectrum analyses help infer very nuanced interactions of a signal's periodicity or seasonal behaviour, when traditional approaches fail or are inconclusive.

Regardless, complex analysis is an area of mathematics that is utterly fascinating, and the prospect of combining its prowess with machine learning is a very exciting thing.

Most of the ML R&D undertaken at Quantaco involves understanding the dynamics of the hospitality industry, and a portion of that includes retail hospitality sales - food and beverages at pubs or other kinds of eateries. Retail sales are known to produce very, very cyclical signals with multiple seasonalities interacting with one another continuously. This kind of behaviour is a very interesting avenue to explore CVNNs with - as we've done - and what follows are running notes from one such experiment at retail sales analysis.

# Foundations
The main thing to keep in mind through all this is that for any complex-valued function $f(z)$, there are always two partial derivatives involved: one for the function with respect to its complex component, $\frac{\partial f}{\partial z}$, and one for the function with respect to its conjugate, $\frac{\partial f}{\partial \bar{z}}$. Both components must be addressed.

## Real and Imaginary Components of $z$
First, some foundation. We need a few concepts in place from complex analysis before we can do this thing. Let's start by (re)defining a few basic properties. To start with, here we have what the real and imaginary components of some complex-valued number $z$ are:

\begin{align*}
  Re(z) &= \mathfrak{R}(z) = \frac{z + \bar{z}}{2}, \\
  Im(z) &= \mathfrak{I}(z) = \frac{z - \bar{z}}{2i}
\end{align*}

## Complex Dot-Product
For vectors in $\mathbb{C}$, the standard definition of the dot product can lead to contradicting results because of how $i$ is defined. For instance, the dot product of a vector with itself can be zero with the vector itself being zero (e.g. $\vec a = \left[ 1, i \right]$). We're going to use the official definition of the complex dot product - "official" because it comes directly from theoretical physics, and those guys definitely know what they're doing:

$$a \cdot b = \sum \limits_i a_i \bar{b}_i$$

Where $a$ and $b$ are both complex vectors. This form preserves the property of positive-definiteness of the dot product, but not symmetry and bilinearity. Additionally, while symmetry is not preserved (i.e., $a_i \bar{b}_i \neq b_i \bar{a}_i$), _conjugate linearity_ is provided (i.e., $a \cdot b = b \cdot a$). We can also depict this in polar form:

$$ z_1 := r_1 e^{i \theta_1}, \quad z_2 := r_2 e^{i \theta_2} $$
$$ z_1 \cdot z_2 = \mathfrak{R}(\bar{z}_1 z_2) = \lvert z_1 \rvert \lvert z_2 \rvert \cos \theta $$

There are numerous ways to naturally derive this definition of the complex dot product. One such straightforward method is as follows:
1. Taking the real component of conjugate multiplication:

   \begin{align*}
     &\mathfrak{R}(\bar{z}_1 z_2) \\
     &\implies \mathfrak{R}((x_1 - i y_1)x_2 + i y_2) \\
     &\implies \mathfrak{R}((x_1 x_2 + y_1 y_2) + i(x_1 y_2 - x_2 y_1)) \\
     &\implies x_1 x_2 + y_1 y_2
   \end{align*}

2. Getting an angle, part 1:

   \begin{align*}
     &\mathfrak{R}(\bar{z}_1 z_2) \\
     &\implies \mathfrak{R}(r_1 e^{i - \theta_1} \cdot r_2 e^{i \theta_2}) \\
     &\implies \mathfrak{R}(r_1 r_2 e^{i \theta_2 - \theta_1}) \\
     &\implies \mathfrak{R}(
        r_1 r_2 (\cos(\theta_2 - \theta_1) + i (\sin(\theta_2 - \theta_1)))
      ) \\
     &\implies r_1 r_2 \cos(\theta_2 - \theta_1)
   \end{align*}

3. Getting an angle, part 2:

   \begin{align*}
     &r_1 r_2 \cos(\theta_2 - \theta_1) \\
     &\implies \lvert z_1 \rvert \lvert z_2 \rvert \cos(\theta_2 - \theta_1) \\
     &\implies \lvert z_1 \rvert \lvert z_2 \rvert \cos(\theta)
   \end{align*}

Where that $\theta$ at the end is $\theta_2 - \theta_1$. Such a definition of the complex dot product allows us to almost trivially implement complex-valued attention. Recall that real-valued attention modules apply a softargmax to a normalised $QK$; because the softargmax inherently works only in $\mathbb{R}$, translating it over to $\mathbb{C}$ is where this definition of the complex dot product shines. Complex-valued $Q \cdot K$ will thus result in a tensor of reals, and all following operations can continue normally.

Looking at measures of similarities, the real part of the exponential (i.e., $\cos(\theta_2 - \theta_1)$) maximises at $1$ when $\theta_2 - \theta_1 = 0$, which occurs when $z_1 = z_2$. It strictly decreases down to a minimum of $-1$ when $\theta_2 - \theta_1 = \pi$, which occurs when $z_1 = -z_2$. This property provides an inherent measure of the similarity between two complex vectors, and also provides a simple notion of what it means to be probabilistic in  as interpreted by traditional attention architectures. While the real dot product gives us the cosine of the angle between two vectors, scaled by their magnitudes, the complex dot product gives us the _cosine of the difference_ in angles between the two vectors, scaled by their magnitudes.

Finally, this definition of the complex dot product also carries over to complex matrices, since matrix multiplication essentially computes the dot product between every one vector in the left matrix and every other vector in the right matrix.

# Wirtinger Derivatives
With this, we can quickly setup the Wirtinger calculus. The derivative for $z$ includes derivatives for both its components, $z$ and its conjugate, $\bar{z}$. These are defined as:

\begin{align*}
  \frac{\partial}{\partial z} &= \frac{\partial}{\partial x} \cdot \frac{\partial x}{\partial z} + \frac{\partial}{\partial y} \cdot \frac{\partial y}{\partial z}, \\
  \frac{\partial}{\partial \bar{z}} &= \frac{\partial}{\partial x} \cdot \frac{\partial x}{\partial \bar{z}} + \frac{\partial}{\partial y} \cdot \frac{\partial y}{\partial \bar{z}}
\end{align*}

## Gradient with respect to $z$
Let's first tackle $z$, then do the conjugate. We can apply our defintions of $\mathfrak{R}(z)$ and $\mathfrak{I}(z)$ to the real and imaginary partials with respect to $z$, $\frac{\partial x}{\partial z}$ and $\frac{\partial y}{\partial z}$:

\begin{align*}
  \frac{\partial x}{\partial z} &= \frac{\partial \left( \frac{z + \bar{z}}{2} \right)}{\partial z} = \frac{1}{2}, \\
  \frac{\partial y}{\partial z} &= \frac{\partial \left( \frac{z - \bar{z}}{2i} \right)}{\partial z} = -\frac{1}{2i} = -\frac{i}{2} \\
  \therefore \frac{\partial}{\partial z} &= \frac{1}{2} \cdot \frac{\partial}{\partial x} - \frac{i}{2} \cdot \frac{\partial}{\partial y} \\
                                         &= \frac{1}{2} \left( \frac{\partial}{\partial x} - i \frac{\partial}{\partial y} \right)
\end{align*}

## Gradient with respect to $\bar{z}$
Now let's do the conjugate. Recall that the conjugate of $z$ is $z$ itself, but with its imaginary part opposite in sign. This means we need to make a small change to our real and imaginary partial gradients with respect to $z$:

\begin{align*}
  \frac{\partial x}{\partial \bar{z}} &= \frac{\partial \left( \frac{z - \bar{z}}{2} \right)}{\partial \bar{z}} = -\frac{1}{2}, \\
  \frac{\partial y}{\partial \bar{z}} &= \frac{\partial \left( \frac{z + \bar{z}}{2i} \right)}{\partial \bar{z}} = \frac{1}{2i} = \frac{i}{2} \\
  \therefore \frac{\partial}{\partial \bar{z}} &= -\frac{1}{2} \cdot \frac{\partial}{\partial x} - \frac{i}{2} \cdot \frac{\partial}{\partial y} \\
                                               &= \frac{1}{2} \left( \frac{\partial}{\partial x} + i \frac{\partial}{\partial y} \right)
\end{align*}

## Putting it together
Now we have our Wirtinger derivatives:

\begin{align*}
  \frac{\partial}{\partial z}       &= \frac{1}{2} \left( \frac{\partial}{\partial x} - i \frac{\partial}{\partial y} \right), \\
  \frac{\partial}{\partial \bar{z}} &= \frac{1}{2} \left( \frac{\partial}{\partial x} + i \frac{\partial}{\partial y} \right)
\end{align*}

There are numerous benefits to using the Wirtinger derivatives, the most important of which - for our purposes - is that everything extends naturally to multivariate functions by defining the cogradient and conjugate cogradient operators, though we will avoid going into that territory here to keep it straightforward.

# Complex Differentiation Rules
## 1. Chain rule
Say we have some complex-valued function $f(z)$, defined as $h(g(z))$:
$$f(z) := h(g(z)) = (h \circ g)(z):  \mathbb{C} \to \mathbb{C}$$

The derivatives of $f$ with respect to $z$ and $\bar{z}$ are then defined as:

\begin{align*}
  \frac{\partial f}{\partial z} &= \frac{h(g(z))}{\partial g(z)} \cdot \frac{g(z)}{\partial z} + \frac{\partial h(g(z))}{\partial g(\bar{z})} \cdot \frac{\partial \bar{g}(z)}{\partial z}, \\
  \frac{\partial f}{\partial \bar{z}} &= \frac{h(g(z))}{\partial g(z)} \cdot \frac{g(z)}{\partial \bar{z}} + \frac{\partial h(g(z))}{\partial g(\bar{z})} \cdot \frac{\partial \bar{g}(z)}{\partial \bar{z}}
\end{align*}

## 2. Product rule
Say we have some complex-valued $f(z)$, defined as $h(z) \cdot g(z)$:
$$f(z) := h(z) \cdot g(z):  \mathbb{C} \to \mathbb{C}$$

The derivatives of $f$ with respect to $z$ and $\bar{z}$ are then defined as:

\begin{align*}
  \frac{\partial f}{\partial z} &= \frac{\partial h(z)}{\partial z} \cdot g(z) + h(z) \cdot \frac{\partial g(z)}{\partial z}, \\
  \frac{\partial f}{\partial \bar{z}} &= \frac{\partial h(z)}{\partial \bar{z}} \cdot g(z) + h(z) \cdot \frac{\partial g(z)}{\partial \bar{z}}
\end{align*}

## 3. Conjugation rules
We also have these two specific conjugation rules to keep in mind:

\begin{align*}
  \overline{\left( \frac{\partial f}{\partial z} \right)} &= \frac{\partial \bar{f}}{\partial \bar{z}}, \\
  \overline{\left( \frac{\partial f}{\partial \bar{z}} \right)} &= \frac{\partial \bar{f}}{\partial z}
\end{align*}

# Differentiating Complex Attention
With all that out of the way, we can slowly dive into differentiating complex attention. A lot of this, as with regular ML, involves repeated application of the complex chain rule - but we also must pay heed to the conjugation rules. They're crucial in helping us tame the behemoth we're about to see.

## Setup
To start with, let's define our complex-valued $Q, K, V$ tensors and a few variables:

\begin{align*}
  Q_{\mathbb{C}} &= \mathbb{C}\text{linear}(Z) = WZ + b \\
  K_{\mathbb{C}} &= \mathbb{C}\text{linear}(Z) = WZ + b \\
  V_{\mathbb{C}} &= \mathbb{C}\text{linear}(Z) = WZ + b \\
  \text{scores}  &= \mathfrak{R}(QK_{\mathbb{C}}^T) = QK_{\mathbb{R}} \\
  \text{weight}  &= \sigma(\text{scores}) \\
  \text{attent}  &= \text{weight} \cdot V_{\mathbb{C}}
\end{align*}

Recall that $\mathfrak{R}(QK_{\mathbb{C}}^T)$ indicates the complex dot product between the tensors $Q_{\mathbb{C}}, K_{\mathbb{C}}$. We can clean up our notation a little bit:

\begin{align*}
              Q &= WZ + b \\
              K &= WZ + b \\
              V &= WZ + b \\
  \text{scores} &= \mathfrak{R}(QK_{\mathbb{C}}^t) \cdot d_K^{-0.5} \\
  \text{activ}  &= \sigma(\mathfrak{R}(QK_{\mathbb{C}}^t) \cdot d_K^{-0.5}) \\
  \text{attent} &= \sigma(\mathfrak{R}(QK_{\mathbb{C}}^t) \cdot d_K^{-0.5}) \cdot V_{\mathbb{C}}
\end{align*}

And now we're in good shape to apply our Wirtinger derivatives from back to front. Note that $d_K$ is the dimension of the tensor $K$, implying $d_K^{-0.5}$ is a constant and can therefore be ignored during differentiation. If you'd like to refresh yourself and/or get a beverage, now would be a good time to do so.

## The System of Equations
There isn't any sufficient way to ease ourselves into this, so let's just have at it. The gradient of $\text{attent}$ is given by:
$$\nabla \text{attent} = \nabla \left[ \text{activ} \cdot V \right] = \nabla \left[ \sigma(\mathfrak{R}(QK_{\mathbb{C}}^t) \cdot d_K^{-0.5}) \cdot V_{\mathbb{C}} \right]$$

We start from here by applying the chain rule to $\left[ \text{activ} \cdot V \right]$ with respect to the input tensor, $Z \in {\mathbb{C}}$:

\begin{align*}
  \frac{\partial \text{attent}}{\partial Z} &= \frac{\partial \text{activ}}{\partial Z} \cdot V + \text{activ} \cdot \frac{\partial V}{\partial Z}, \\
  \frac{\partial \text{attent}}{\partial \bar{Z}} &= \frac{\partial \text{activ}}{\partial \bar{Z}} \cdot V + \text{activ} \cdot \frac{\partial V}{\partial \bar{Z}}
\end{align*}

Now we dig into $\text{activ}$ with respect to $Z$, and paying very close attention to what is being conjugated and where:

\begin{align*}
  \frac{\partial \text{activ}}{\partial Z} &=
          \frac{\partial \text{activ}}{\partial \text{scores}}
    \cdot \frac{\partial \text{scores}}{\partial Z}
    +     \frac{\partial \text{activ}}{\partial \text{scores}(\bar{Z})}
    \cdot \frac{\partial \overline{\text{scores}}(Z)}{\partial Z}, \\
  \frac{\partial \text{activ}}{\partial \bar{Z}} &=
          \frac{\partial \text{activ}}{\partial \text{scores}}
    \cdot \frac{\partial \text{scores}}{\partial \bar{Z}}
    +     \frac{\partial \text{activ}}{\partial \text{scores}(\bar{Z})}
    \cdot \frac{\partial \overline{\text{scores}}(Z)}{\partial \bar{Z}}
\end{align*}

Focusing on the gradient of $\text{activ}$ with respect to $\text{scores}$, something to note is that this gradient is simply the regular gradient of softargmax $\in \mathbb{R}$, no conjugates, owing to the way the complex dot product works. This gradient thus loses its conjugate component, and becomes:
$$\frac{\partial \text{activ}}{\partial \text{scores}} = \frac{\partial \sigma(\mathfrak{R}(QK_{\mathbb{C}}^t) \cdot d_K^{-0.5})}{\partial \mathfrak{R}(QK_{\mathbb{C}}^t)}$$

Let's look at the next bit in line, the gradient of $\text{scores}$ with respect to $Z$:

\begin{align*}
  \frac{\partial \text{scores}}{\partial Z} &=
          \frac{\partial \mathfrak{R}(QK^T)}{\partial QK^T}
    \cdot \frac{\partial QK^T}{\partial Z}
    +     \frac{\partial \mathfrak{R}(QK^T)}{\partial QK^T(\bar{Z})}
    \cdot \frac{\partial \overline{QK^T}(Z)}{\partial Z}, \\
  \frac{\partial \text{scores}}{\partial \bar{Z}} &=
          \frac{\partial \mathfrak{R}(QK^T)}{\partial QK^T}
    \cdot \frac{\partial QK^T}{\partial \bar{Z}}
    +     \frac{\partial \mathfrak{R}(QK^T)}{\partial QK^T(\bar{Z})}
    \cdot \frac{\partial \overline{QK^T}(Z)}{\partial \bar{Z}}
\end{align*}

We're very close to the crux of all this. We'll get back to the gradient of $\mathfrak{R}(QK^T)$ with respect to $QK^T$ later on, for now let's look at the gradient of $QK^T$ with respect to $Z$:

\begin{align*}
  \frac{\partial QK^T}{\partial Z} &= \frac{\partial Q}{\partial Z} \cdot K + Q \cdot \frac{\partial K}{\partial Z}, \\
  \frac{\partial QK^T}{\partial \bar{Z}} &= \frac{\partial Q}{\partial \bar{Z}} \cdot K + Q \cdot \frac{\partial K}{\partial \bar{Z}}
\end{align*}

## The crux of all this
Since all of $Q, K, V$ are simple linear layers, the gradient of a generic complex-valued linear layer $F = WZ + b$ - purely with respect to the input tensor, $Z$ - is applicable to all of them. The bias, $b_{\mathbb{C}}$, is differentiated with respect to the loss function as in the real scenario; there isn't any dependence on $Z \in \mathbb{C}$:
$$\nabla F = \nabla WZ + b$$

\begin{align*}
  \implies \frac{\partial F}{\partial Z} &= \frac{1}{2} \left( \frac{\partial W(X + iY)}{\partial X} - i \frac{\partial W(X + iY)}{\partial Y} \right) \\
                                         &= \frac{1}{2} (W - i(iW)) \\
                                         &= W \\
  \frac{\partial F}{\partial \bar{Z}} &= \frac{1}{2} \left( \frac{\partial W(X + iY)}{\partial X} + i \frac{\partial W(X + iY)}{\partial Y} \right) \\
                                      &= \frac{1}{2} (W + i(iW)) \\
                                      &= 0
\end{align*}

The gradient of the complex-valued bias - again, only relative to the loss function, $L$ - is:

\begin{align*}
  \implies \frac{\partial F}{\partial b} &= \frac{1}{2} \left( \frac{\partial}{\partial X} - i \frac{\partial}{\partial Y} \right) \cdot L_{\mathbb{R}}, \\
  \implies \frac{\partial F}{\partial \bar{b}} &= \frac{1}{2} \left( \frac{\partial}{\partial X} + i \frac{\partial}{\partial Y} \right) \cdot L_{\mathbb{R}}
\end{align*}

This is rather pleasant, because it implies that a complex-valued linear layer satisfies the CR equations, and it lets us navigate most of our to-be-solved gradients right to zero. We can substitute the gradient of $F$ to find $Q, K, V$ with respect to $Z$:

\begin{align*}
  \frac{\partial Q}{\partial Z} &= W_Q, \text{ and } \frac{\partial Q}{\partial \bar{Z}} = 0 \\
  \frac{\partial K}{\partial Z} &= W_K, \text{ and } \frac{\partial K}{\partial \bar{Z}} = 0 \\
  \frac{\partial V}{\partial Z} &= W_V, \text{ and } \frac{\partial V}{\partial \bar{Z}} = 0
\end{align*}

It gets much simpler now!

## Back to the system
Past the crux of all this, we can recursively start to substitute our complex-linear gradient back into whatever we have so far. The conjugation rules play a very important role here. Starting slow, we begin with:

\begin{align*}
  \frac{\partial QK^T}{\partial Z} &= W_Q K + QW_K^T, \\
  \frac{\partial QK^T}{\partial \bar{Z}} &= 0K + Q0 = 0
\end{align*}

From our conjugation rules, we can infer the following:

\begin{align*}
  \frac{\partial \overline{QK^T}}{\partial Z} &= 0 = \overline{\left( \frac{\partial QK^T}{\partial \bar{Z}} \right)} = \bar{0} = 0, \\
  \frac{\partial \overline{QK^T}}{\partial \bar{Z}} &= \overline{\left( \frac{\partial QK^T}{\partial Z} \right)} = \overline{W_Q K + QW_K^T} = \overline{W_Q K} + \overline{QW_K^T}
\end{align*}

Which we can now use to finally differentiate $scores$:

\begin{align*}
  \frac{\partial \text{scores}}{\partial Z} &= \frac{1}{2} \cdot W_Q K + QW_K^T + \frac{1}{2} \cdot 0, \\
  \frac{\partial \text{scores}}{\partial \bar{Z}} &= \frac{1}{2} \cdot 0 + \frac{1}{2} \cdot \overline{W_Q K} + \overline{QW_K^T}
\end{align*}

And simplify into:

\begin{align*}
  \frac{\partial \text{scores}}{\partial Z} &= \frac{1}{2} \cdot W_Q K + QW_K^T, \\
  \frac{\partial \text{scores}}{\partial \bar{Z}} &= \frac{1}{2} \cdot \overline{W_Q K} + \overline{QW_K^T}
\end{align*}

Notice how our conjugate gradient is now non-zero.

## The gradient of softargmax $\in \mathbb{R}$
Before proceeding, at this point it is highly recommended to revise the derivation of the gradient of softargman $\in \mathbb{R}$. This is a very well known derivation and involves using logarithms to handle the exponents, and won't be included here for pithiness. Just know that the final result of this gradient, for our purposes and with our variables, is:
$$\frac{\partial \text{activ}}{\partial Z_j} = \text{activ} \cdot \left( 1 \lbrace i = j \rbrace - \text{activ}_j \right)$$

Where $j$ is the $j$-th element of the input tensor to the softargmax function. Now going back to system, we have:

\begin{align*}
  \frac{\partial \text{activ}}{\partial \text{scores}} &= \text{activ} \cdot \left( 1 \lbrace i = j \rbrace - \text{activ}_j \right), \\
  \frac{\partial \text{activ}}{\partial \overline{\text{scores}}} &= 0
\end{align*}

Our conjugate gradient here is zero because our softargmax only operates in $\mathbb{R}$! This now gives us:

\begin{align*}
  \frac{\partial \text{activ}}{\partial Z} &= \text{activ} \cdot \left( 1 \lbrace i = j \rbrace - \text{activ}_j \right) \cdot \frac{1}{2} \cdot W_Q K + QW_K^T + 0, \\
  \frac{\partial \text{activ}}{\partial \bar{Z}} &= \text{activ} \cdot \left( 1 \lbrace i = j \rbrace - \text{activ}_j \right) \cdot \frac{1}{2} \cdot \overline{W_Q K} + \overline{QW_K^T} + 0
\end{align*}

## Final steps
And finally, substituting our gradients for $\text{activ}$ back into our first expression gives us:

\begin{align*}
  \frac{\partial \text{attent}}{\partial Z} &=
          \text{activ}
    \cdot \left( 1 \lbrace i = j \rbrace - \text{activ}_j \right)
    \cdot \frac{1}{2}
    \cdot \left( W_Q K + QW_K^T \right)
    \cdot V
    +     \text{activ}
    \cdot W_V, \\
  \frac{\partial \text{attent}}{\partial \bar{Z}} &=
          \text{activ}
    \cdot \left( 1 \lbrace i = j \rbrace - \text{activ}_j \right)
    \cdot \frac{1}{2}
    \cdot \left( \overline{W_Q K} + \overline{QW_K^T} \right)
    \cdot V
    +     \text{activ}
    \cdot 0
\end{align*}

And we're done!

## Gradient Derived
Cleaning up, we have our final solution(s):

\begin{align*}
  \frac{\partial \text{attent}}{\partial Z} &=
          \text{activ}
    \cdot \left( 1 \lbrace i = j \rbrace - \text{activ}_j \right)
    \cdot \frac{1}{2}
    \cdot \left( W_Q K + QW_K^T \right)
    \cdot V
        + \text{activ}
    \cdot W_V, \\
  \frac{\partial \text{attent}}{\partial \bar{Z}} &=
          \text{activ}
    \cdot \left( 1 \lbrace i = j \rbrace - \text{activ}_j \right)
    cdot \frac{1}{2}
    \cdot \left( \overline{W_Q K} + \overline{QW_K^T} \right)
    \cdot V
\end{align*}

# Closing Thoughts
So what was all of this for, and what can we infer from this exercise? Well for starters, the most obvious inference is that complex-valued attention is not holomorphic. Recall that a complex map is holomorphic at $Z$ _IFF_ the conjugate Wirtinger derivative vanishes there, i.e.:
$$\frac{\partial f}{\partial \bar{Z}} = 0$$

Our conjugate attention gradient is non-zero because complex conjugation cannot disappear through the real-valued softargmax. In other words, the attention layer is not complex-analytic, and this means it simply acts a regular real-valued 2D map, $\mathbb{R}^2 \to \mathbb{R}^2$, where complex-analytic techniques like contour integrals or residue calculus are not viable.

Secondly, conformal maps are not preserved for the same reason. In other words, local angles and shapes are not preserved in this layer because it isn't holomorphic, and this is a big deal. The reason why this happens is because of this expression:
$$\frac{\partial QK^T}{\partial Z} = W_Q K + QW_K^T$$

Since the weights of $Q$, i.e. $W_Q$ is a fixed tensor, multiplying it with the current key vector $K$ (a column) produces the Jacobian outer product, a matrix with rank $\le 1$ since it's built from one column and one row. Likewise with $QW_K^T$, and for all key vectors $\in K$. The addition of these two dyads causes the unit circle in $Z$ space to collapse into an ellipse and potentially, eventually a line. This is the same issue addressed by Chiheb Trabelsi in the paper Complex Batch Normalisation, so complex batch norm becomes all the more important.

Third, since the classical Cauchy integral theorem no longer applies due to non-holomorphic behaviour, physical interpretation in $C$ is not possible. This does not render the layer totally uninterpretable; as mentioned, it just means inference must occur in $\mathbb{R}^2$, i.e. in real 2D geometry.

Finally, as is the case with CVNNs in general, the optimisaton of this layer implies the "learning" of phase information from the input signal. Because we're using complex numbers here and the imaginary component encodes phasic information, the attention layer modulates attention not just on the basis of magnitude similarity but also on the basis of phase offsets between features. This expressiveness is invaluable when dealing with image data or other signals over time that are highly cyclical, periodic, or that have an implicit dependence on phase.

Regardless, even though the attention layer in $\mathbb{C}$ is not holomorphic, it doesn't make too much of a difference. Computationally, most autodiff libraries treat the real and imaginary components separately, and modern CVNN applications willingly break holomorphicity to keep phase and magnitude pathways independent.

## What Does This All Mean for Retail Sales KDD?
Basically, we lose a lot of theoretical power as far as attention-based CVNNs are concerned, but we do gain the ability to modulate attention on the basis of signal magnitude and phase, and we get to actually leverage phase in a very neat way. Consider the general analytical problem at Quantaco: many hospitality vendors means many hospitality sales, sometimes to the order of greater than 1:1 because of the variety of products offered by a specific vendor, all separated by geography and time. Because complex parameters encode magnitude and phase in one representation, it is very easy to exploit periodicity _and_ phasic behaviour in one model.

For a real example, two pubs one street apart may show the same 24 hour pattern but shifted by 30 minutes - identical frequency, different phase. It is theoretically possible for a model to "learn" to pay attention to the sales of outlet B if its sales align with outlet A, for example. Pubs from the same parent company might show coherent sales across geographies on holidays, implying phases aligned by holiday whilst frequencies can differ; in fact for another real example, this extreme correlation behaviour is exacerbated across all pubs during black-swan events like ANZAC day or a F1 race screening - much like the markets moving in sync during a major event. Another aspect is that periodicity is traditionally represented with Fourier sine and cosine sums, which require two explicit columns or parameters in a model to handle; CVNNs would manipulate this _and_ the phase between them in one single update step. Finally, downstream analysis of magnitude and phase offsets for specific outlets can itself provide a great insight into industry dynamics (e.g. "why did sales shift from time $\tau_1$ to $\tau_N$?", or "what's influencing this disparity between both of my outlets?").

Evidently, there is much more merit in the multivariate setting than the univariate setting where relativity between time series carries a signal, even though univariate settings also benefit. And equally evidently, there is much excitement to go around in complex-valued machine learning.

---

# References and Further Reading
- [Dot Product of Complex Numbers](https://proofwiki.org/wiki/Definition:Dot_Product#Complex_Numbers), retrieved 22 April, 2025, ProofWiki.
- [Wirtinger Derivatives](https://en.wikipedia.org/wiki/Wirtinger_derivatives), retrieved 22 April, 2025, Wikipedia.
- Ken Kreutz-Delgado, ["The Complex Gradient Operator and the CR-Calculus"](https://arxiv.org/abs/0906.4835), 26 June,2009, ArXiv.
- Thomas Kurbiel, ["Derivative of the Softmax Function and the Categorical Cross-Entropy Loss"](https://medium.com/data-science/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1), 23 April, 2021, Medium.
- PyTorch, ["Autograd for Complex Numbers"](https://pytorch.org/docs/stable/notes/autograd.html#autograd-for-complex-numbers), retrieved 22 April, 2025, PyTorch.
- Joshua Bassey, ["A Survey of Complex-Valued Neural Networks"](https://arxiv.org/abs/2101.12249), 28 January, 2021, ArXiv.
- Chiheb Trabelsi, ["Deep Complex Networks"](https://arxiv.org/abs/1705.09792), 25 February, 2018, ArXiv.
- Hyun-Woong Cho, ["Complex-Valued Channel Attention and Application in Ego-Velocity Estimation With Automotive Radar"](https://ieeexplore.ieee.org/document/9335579), 2021, IEEE.
- Kun Yi, ["Frequency-domain MLPs are More Effective Learners in Time Series Forecasting (FreTS)"](https://arxiv.org/abs/2311.06184), 10 November 2023, ArXiv.
- Josiah W. Smith, ["Complex-Valued Neural Networks for Data-Driven Signal Processing and Signal Understanding"](https://arxiv.org/abs/2309.07948), 14 September 2023, ArXiv.
- thisiszilff, ["Understanding and Coding the Self-Attention Mechanism"](https://news.ycombinator.com/item?id=34748584), 11 February 2023, YCombinator.
