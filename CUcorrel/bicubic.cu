// w0, w1, w2, and w3 are the four cubic B-spline basis functions
__device__ float w0(float a)
{
    return (1.0f/6.0f)*(a*(a*(-a + 3.0f) - 3.0f) + 1.0f);
}

__device__ float w1(float a)
{
    return (1.0f/6.0f)*(a*a*(3.0f*a - 6.0f) + 4.0f);
}

__device__ float w2(float a)
{
    return (1.0f/6.0f)*(a*(a*(-3.0f*a + 3.0f) + 3.0f) + 1.0f);
}

__device__ float w3(float a)
{
    return (1.0f/6.0f)*(a*a*a);
}

// g0 and g1 are the two amplitude functions
__device__ float g0(float a)
{
    return w0(a) + w1(a);
}

__device__ float g1(float a)
{
    return w2(a) + w3(a);
}

// h0 and h1 are the two offset functions
__device__ float h0(float a)
{
    // note +0.5 offset to compensate for CUDA linear filtering convention
    return -1.0f + w1(a) / (w0(a) + w1(a)) + 0.5f;
}
__device__ float h1(float a)
{
    return 1.0f + w3(a) / (w2(a) + w3(a)) + 0.5f;
}

__device__ float interpBicubic(cudaTextureObject_t tex, float x, float y, float w, float h)
{
    //x -= 0.5f;
    //y -= 0.5f;
    float px = floor(x);
    float py = floor(y);
    float fx = x - px;
    float fy = y - py;

    // note: we could store these functions in a lookup table texture, but maths is cheap
    float g0x = g0(fx);
    float g1x = g1(fx);
    float h0x = h0(fx);
    float h1x = h1(fx);
    float h0y = h0(fy);
    float h1y = h1(fy);

    float r = g0(fy) * (g0x * tex2D<float>(tex, (px + h0x)/w, (py + h0y)/h)   +
                    g1x * tex2D<float>(tex, (px + h1x)/w, (py + h0y)/h)) +
          g1(fy) * (g0x * tex2D<float>(tex, (px + h0x)/w, (py + h1y)/h)   +
                    g1x * tex2D<float>(tex, (px + h1x)/w, (py + h1y)/h));
    return r;
}

