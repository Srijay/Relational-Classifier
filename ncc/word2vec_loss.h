#ifndef EXPERIMENTAL_USERS_SOUMENC_RECTANGLE_WORD2VEC_LOSS_H_
#define EXPERIMENTAL_USERS_SOUMENC_RECTANGLE_WORD2VEC_LOSS_H_

#include <vector>
#include <cmath>

namespace ncc {

/**
 * Provides numerically safe implementation of functions useful in expressing
 * losses and gradients.
 */
template <typename Rn>
class LogitUtils {
 public:

  static Rn dot(const Rn *const veca, const Rn *const vecb, int ndim) {
    Rn ans = 0;
    for (int dim = 0; dim < ndim; ++dim) {
      ans += veca[dim] * vecb[dim];
    }
    return ans;
  }

  // Returns log(1 + exp(a)) = a + log(exp(-a) + 1)
  static Rn log1pexp(Rn a) {
    if (a < 0) return std::log(1. + std::exp(a));
    else return a + std::log(std::exp(-a) + 1);
  }

  static Rn logit(Rn a) {
    if (a > 0) {
      return 1. / (1. + std::exp(-a));
    }
    else {
      const Rn tmp = std::exp(a);
      return tmp / (1. + tmp);
    }
  }

  static Rn logitGrad(Rn x) {
    if (x < 0) {
      const Rn expx = std::exp(x);
      return expx / (1. + expx) / (1. + expx);
    }
    else {
      const Rn expnx = std::exp(-x);
      return expnx / (1. + expnx) / (1. + expnx);
    }
  }

  static void op_dot(Rn *fvec, Rn *cvec, Rn *out, Rn *fgrad, Rn *cgrad, int ndim) {
    *out = 0;
    for (int dim = 0; dim < ndim; ++dim) {
      *out += fvec[dim] * cvec[dim];
      fgrad[dim] = cvec[dim];
      cgrad[dim] = fvec[dim];
    }
  }

  static void op_loss_logit(Rn in, Rn label, Rn *out, Rn *grad) {
    *out = log1pexp(in) - in * label;
    *grad = logit(in) - label;
  }

  static void op_loss_logit_dot(Rn *fvec, Rn *cvec, Rn label,
                                Rn *out, Rn *fgrad, Rn *cgrad, int ndim)
  {
    Rn dot;
    op_dot(fvec, cvec, &dot, fgrad, cgrad, ndim);
    Rn loss, loss_grad;
    op_loss_logit(dot, label, &loss, &loss_grad);
    *out = loss;
    for (int dim = 0; dim < ndim; ++dim) {
      fgrad[dim] *= loss_grad;
      cgrad[dim] *= loss_grad;
    }
  }

  static void op_loss_legacy(Rn *fvec, Rn *cvec, Rn label,
                             Rn *out, Rn *fgrad, Rn *cgrad, int ndim) {
    Rn dot = 0;
    for (int dim = 0; dim < ndim; ++dim) {
      dot += fvec[dim] * cvec[dim];
    }
    *out = log1pexp(dot) - label * dot;
    CHECK(!std::isnan(*out)) << dot;
    const Rn pred = logit(dot), mult = (pred - label);
    for (int dim = 0; dim < ndim; ++dim) {
      fgrad[dim] = cvec[dim] * mult;
      cgrad[dim] = fvec[dim] * mult;
    }
  }

#if 0
  static void op_loss_dot(Rn dot, Rn label, Rn *out, Rn *grad) {
    *out = (1. - label) * dot + log1pexp(-dot);
    if (dot < 0) {
      *grad = (1. - label) + ( -1. / ( 1. + std::exp(dot) ) );
    }
    else {
      const Rn nexp = std::exp(-dot);
      *grad = (1. - label) + ( -nexp / (nexp + 1.) );
    }
  }
#endif

  static void op_dist2(Rn *fvec, Rn *cvec, Rn *out, Rn *fgrad, Rn *cgrad, int ndim) {
    *out = 0;
    for (int dim = 0; dim < ndim; ++dim) {
      const Rn diff = fvec[dim] - cvec[dim];
      *out += (diff * diff);
      fgrad[dim] = 2. * (fvec[dim] - cvec[dim]);
      cgrad[dim] = 2. * (cvec[dim] - fvec[dim]);
    }
  }

  static void op_sub(Rn a, Rn b, Rn *out, Rn *agrad, Rn *bgrad) {
    *out = a - b;
    *agrad = 1.;
    *bgrad = -1.;
  }

  static void op_loss_logit_dist2(Rn *fvec, Rn *cvec, Rn rad, Rn label,
                                  Rn *out, Rn *fgrad, Rn *cgrad, Rn *radgrad,
                                  int ndim)
  {
    Rn dist2;
    op_dist2(fvec, cvec, &dist2, fgrad, cgrad, ndim);
    Rn sub, sub_grad_rad, sub_grad_dist2;
    op_sub(rad, dist2, &sub, &sub_grad_rad, &sub_grad_dist2);
    Rn loss, loss_grad;
    op_loss_logit(sub, label, &loss, &loss_grad);
    *out = loss;
    *radgrad = loss_grad * sub_grad_rad;
    for (int dim = 0; dim < ndim; ++dim) {
      fgrad[dim] = fgrad[dim] * loss_grad * sub_grad_dist2;
      cgrad[dim] = cgrad[dim] * loss_grad * sub_grad_dist2;
    }
  }

};


/**
 * Mimics Mikolov update but for only one pair and with separate
 * buffered gradients.
 */
template <typename Rn>
class LogitDotUpdater {
 protected:
  const int ndim, vocabSize;
  std::vector<Rn> fgrad, cgrad;

 public:
  LogitDotUpdater() = delete;

  LogitDotUpdater(int ndim_, int vocabSize_)
  : ndim(ndim_), vocabSize(vocabSize_) {
    fgrad.clear(); fgrad.resize(ndim, 0);
    cgrad.clear(); cgrad.resize(ndim, 0);
  }

  Rn trainPair(Rn *fvec, Rn *cvec, Rn label, Rn step) {
    Rn loss;
    LogitUtils<Rn>::op_loss_logit_dot(fvec, cvec, label,
                                      &loss, fgrad.data(), cgrad.data(), ndim);
    for (int dim=0; dim<ndim; ++dim) {
      fvec[dim] -= step * fgrad[dim];
      cvec[dim] -= step * cgrad[dim];
    }
    return loss;
  }
};


/**
 * Pr(cooccur|u, v) = logit(radius - dist2(u,v)).
 */
template <typename Rn>
class LogitNdist2Updater {
 protected:
  const int ndim, vocabSize;
  std::vector<Rn> fgrad, cgrad;
  Rn rgrad;

 public:
  LogitNdist2Updater() = delete;

  LogitNdist2Updater(int ndim_, int vocabSize_)
  : ndim(ndim_), vocabSize(vocabSize_) {
    fgrad.clear(); fgrad.resize(ndim, 0);
    cgrad.clear(); cgrad.resize(ndim, 0);
  }

  Rn trainPair(Rn *fvec, Rn *cvec, Rn *radius, Rn label, Rn step) {
    Rn loss;
    LogitUtils<Rn>::op_loss_logit_dist2(fvec, cvec, *radius, label,
                                        &loss, fgrad.data(), cgrad.data(), &rgrad,
                                        ndim);
    *radius -= step * rgrad;
    for (int dim =0; dim < ndim; ++dim) {
      fvec[dim] -= step * fgrad[dim];
      cvec[dim] -= step * cgrad[dim];
    }
    return loss;
  }
};



}   // namespace ncc

#endif  // EXPERIMENTAL_USERS_SOUMENC_RECTANGLE_WORD2VEC_LOSS_H_
