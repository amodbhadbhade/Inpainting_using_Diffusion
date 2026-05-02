## Intro

Hello everyone. Today we are going to do a deep dive into diffusion models. This powerful family of models are widely used in most of the most popular multimodal LLMs today including ChatGPT, Gemini, etc. And it is a key technical driver behind the latest wave of generative AI applications you are seeing across the tech industry. For example, Midjourney, Stable Diffusion, etc.

We will be covering topics including DDPM, DDIM, guided diffusion, latent diffusion, U-Net architecture, and more. I will not only focus on the high levels but also the training and sampling details, model architectures, and underlying math. Let's dive into it.

Before we start, I want to say this talk is relatively math-heavy. However, it won't affect your ability to understand the high-level mechanism. It also won't stop you from experimenting. So if you don't understand, don't worry about it.

*   **DDPM** (Denoising Diffusion Probabilistic Model) is the original diffusion model.
*   **DDIM** (Denoising Diffusion Implicit Models) and **LDM** (Latent Diffusion Model, or Stable Diffusion) are improvements built on top of the original DDPM.
*   DDIM speeds up the sampling process, and LDM improves efficiency further by moving into a latent space.
*   **U-Net** is the model architecture used in all diffusion models.

I will not cover some topics in detail if they were covered in previous videos, such as Autoencoder Deep Dive or Generative Image Overview. These include CNN basics, ELBO, KL divergence, and reparameterization. So just try to enjoy and have fun, just like you enjoy a funny meme.

## Diffusion Background

Let’s start with diffusion at a high level. Say we have $256 \times 256$ pixels to represent an image. It's a square image, and each pixel has three color channels (RGB). We now have a space of $256 \times 256 \times 3$ to represent images. Within this space, each element needs specific values to form a meaningful image.

Say we want to form a sunflower drawing; all the possible combinations of sunflower drawing pixels form a space or distribution (represented here as a yellow cloud). If we can train a model with the capability to traverse back to that cloud from any random point in the blue box (the total pixel space), we will be able to generate images of sunflowers. If we extend this beyond one sunflower to more distributions representing more kinds of images, we can generate anything.

The idea of diffusion is first to have a forward process, where a point in the "meaningful" cloud slowly moves out to become random noise. Then, we use a machine learning model to learn the reverse process to traverse back.

This idea is actually inspired by physics. In physics, diffusion describes the net movement of atoms or molecules from a region of higher concentration to lower concentration, driven by a concentration gradient and random motion (Brownian motion). A drop of ink spreading in water is a classic example. The original DDPM paper studied non-equilibrium thermodynamics; the process of gradually adding noise and then reversing it is very similar to a system moving away from and then back towards an ordered state.

## Forward Diffusion

In the forward diffusion process, an image is transformed into pure noise (usually a standard normal distribution) in $T$ steps. This is done via a Markov Chain where the image at timestamp $t$ maps to its subsequent state at $t+1$. Each step depends only on the previous one.

The formula representing the transition from $x_{t-1}$ to $x_t$ is:
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I})$$

Gaussian noise is added at each step, but it's not constant; the rate is determined by a scheduler ($\beta_t$).

### The Reparameterization Trick:

Normally, to get a sample at $x_t$, we would have to iterate from $0$ to $t$, which is slow. Instead, we use the reparameterization trick. Let $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$. We can directly express $x_t$ in terms of the original image $x_0$:
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$

Where $\epsilon \sim \mathcal{N}(0, \mathbf{I})$. This allows us to sample any noise level $x_t$ instantly without the Markov Chain. Schedulers like linear or cosine significantly impact how clearly the image generates over time.

## Reverse Diffusion

This is where the machine learning happens. The goal is to start from random noise ($x_T$) and iteratively remove noise to get $x_0$.

While we know $q(x_t | x_{t-1})$, the true reverse $q(x_{t-1} | x_t)$ is intractable to compute because it requires knowing the entire distribution of all possible images. Instead, we approximate it with a learned model $p_\theta(x_{t-1} | x_t)$, usually defined as:
$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

Insight: While $q(x_{t-1} | x_t)$ is intractable, the posterior conditioned on $x_0$—$q(x_{t-1} | x_t, x_0)$—is tractable. It is also a Gaussian distribution. By substituting $x_0$ into the formula, we find the mean is a function of $x_t$ and the noise $\epsilon_t$.

The original DDPM authors chose to have the model predict the noise ($\epsilon_\theta$) at each step rather than the mean. The variance ($\Sigma_\theta$) is often kept as a fixed constant ($\beta_t$).

## Loss Function

To train the network, we maximize the log-likelihood of the generated samples. Since the direct integral is intractable, we use the Variational Lower Bound (VLB) or ELBO, similar to a VAE.

After mathematical simplification, the complex VLB reduces to a very simple Mean Squared Error (MSE) loss:
$$\mathcal{L}_{simple} = E_{x_0, \epsilon, t} [ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 ]$$

In short: the training involves adding noise to an image (forward) and training the U-Net to predict exactly what that noise was so it can be subtracted (reverse).

## Why Predict Noise?

*   **Simplifies Math:** It makes the loss function a simple MSE.
*   **Better Scaling:** Noise always has a consistent scale (standard normal), whereas image pixels vary wildly. This makes training more stable.
*   **Empirical Success:** It simply works better in practice.

## Training & Sampling Process

### Training:
*   Sample an image $x_0$ and a random time $t$.
*   Sample random noise $\epsilon$.
*   Calculate $x_t$ using the closed-form formula.
*   Feed $x_t$ and $t$ (as an embedding) into the U-Net.
*   Update model weights by comparing predicted noise to the actual noise $\epsilon$.

### Sampling:
*   Start with pure Gaussian noise $x_T$.
*   For $T$ steps (e.g., 1000):
    *   Predict the noise $\epsilon_\theta$.
    *   Remove a fraction of that noise to get $x_{t-1}$.
*   Repeat until you reach $x_0$.

## DDIM (Denoising Diffusion Implicit Models)

DDPM is slow because it requires 1,000 steps. DDIM allows for much faster sampling by making the process non-Markovian.

The training objective is identical to DDPM, but the sampling allows you to skip steps. For example, you can sample in 50 steps instead of 1,000 using a uniform partition. DDIM can also be deterministic (by setting the variance $\sigma_t$ to zero), meaning the same starting noise will always produce the same image.

## U-Net Architecture

The U-Net is the core "brain" of the diffusion model. Its output size equals its input size. Key components include:

*   **Residual Blocks:** To preserve spatial details.
*   **Timestep Embeddings:** Sinusoidal embeddings (like Transformers) to tell the model which "noise level" it is currently looking at.
*   **Multi-head Attention:** To capture global relationships between pixels.
*   **Group Normalization:** Preferred over Layer Norm because it is more efficient for varying image resolutions and independent of batch size.

## Guided Diffusion

How do we control what the model generates (e.g., "generate a 4")? In Guided Diffusion, we feed an extra embedding (like a text query) into the U-Net. During training, the self-attention blocks learn to associate the text context with the image structure.

## Stable Diffusion (Latent Diffusion Models)

Pixel-space diffusion is computationally expensive. LDM performs diffusion in a lower-dimensional latent space.

*   **Encoder:** Shrinks the image into a smaller latent representation ($z$).
*   **Diffusion:** Noise is added/removed within this small latent space.
*   **Decoder:** Converts the final latent back into a full-sized image.

Stable Diffusion uses Cross-Attention to connect text prompts to the image. Text is converted to embeddings (using CLIP or BERT), and these act as the "Keys" and "Values" in the U-Net's attention layers, while the image latents act as the "Queries."

### VQ-VAE (Vector Quantized VAE):
Stable Diffusion often uses a VQ-VAE for the latent space. Unlike a standard VAE, it uses a discrete codebook (a dictionary of vectors). This avoids "posterior collapse" and results in more meaningful, structured latent codes.

## Conclusion

In summary, we've covered the physics-inspired roots of DDPM, the efficiency of DDIM, the architecture of the U-Net, and how Stable Diffusion (LDM) uses latent spaces and cross-attention to generate high-quality images from text.
