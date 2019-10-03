#pragma once
#include <cmath>

// This is free and unencumbered software released into the public domain.

/* Approximate Bessel Function from
"Precise analytic approximations for the Bessel function J1(x)" by Fernando Maass and Pablo Martin
Results in Physics, https://doi.org/10.1016/j.rinp.2018.01.071 */
template<typename T>
auto besselJ1(T x) -> decltype(1.f*x) {
    /* Original equations
       (0.8660 + 0.1601 xx) sin(x)              -x (0.3718 + 0.1007 xx) cos(x)
    ------------------------------------  +  ------------------------------------
    (1 + 0.3489 xx)(1 + 0.4181 xx)^(1/4)     (1 + 0.3489 xx)(1 + 0.4181 xx)^(3/4)
    The procedure here comes from simplifying it in Mathematica
    */
    if(x == 0) return 0;
    auto x2 = x*x;
    auto a = 0.458871f*sqrt(1 + 0.4181f*x2)*(x2 + 5.40912f)*sin(x);
    auto b = x*(1.06563f + 0.288621f*x2)*cos(x);
    auto c = pow(1 + 0.4181f*x2, 0.75f)*(2.86615f + x2);
    return (a-b)/c;
}
