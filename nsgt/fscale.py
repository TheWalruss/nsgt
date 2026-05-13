# -*- coding: utf-8

"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011-2015
http://grrrr.org/nsgt

Austrian Research Institute for Artificial Intelligence (OFAI)
AudioMiner project, supported by Vienna Science and Technology Fund (WWTF)
"""

import numpy as np

class Scale:
    dbnd = 1.e-8
    
    def __init__(self, bnds):
        self.bnds = bnds
        
    def __len__(self):
        return self.bnds
    
    def Q(self, bnd=None):
        # numerical differentiation (if self.Q not defined by sub-class)
        if bnd is None:
            bnd = np.arange(self.bnds)
        return self.F(bnd)*self.dbnd/(self.F(bnd+self.dbnd)-self.F(bnd-self.dbnd))
    
    def __call__(self):
        f = np.array([self.F(b) for b in range(self.bnds)],dtype=float)
        q = np.array([self.Q(b) for b in range(self.bnds)],dtype=float)
        return f,q


class OctScale(Scale):
    def __init__(self, fmin, fmax, bpo, beyond=0):
        """
        @param fmin: minimum frequency (Hz)
        @param fmax: maximum frequency (Hz)
        @param bpo: bands per octave (int)
        @param beyond: number of frequency bands below fmin and above fmax (int)
        """
        lfmin = np.log2(fmin)
        lfmax = np.log2(fmax)
        bnds = int(np.ceil((lfmax-lfmin)*bpo))+1
        Scale.__init__(self, bnds+beyond*2)
        odiv = (lfmax-lfmin)/(bnds-1)
        lfmin_ = lfmin-odiv*beyond
        lfmax_ = lfmax+odiv*beyond
        self.fmin = 2**lfmin_
        self.fmax = 2**lfmax_
        self.pow2n = 2**odiv
        self.q = np.sqrt(self.pow2n)/(self.pow2n-1.)/2.
        
    def F(self, bnd=None):
        return self.fmin*self.pow2n**(bnd if bnd is not None else np.arange(self.bnds))
    
    def Q(self, bnd=None):
        return self.q


class LogScale(Scale):
    def __init__(self, fmin, fmax, bnds, beyond=0):
        """
        @param fmin: minimum frequency (Hz)
        @param fmax: maximum frequency (Hz)
        @param bnds: number of frequency bands (int)
        @param beyond: number of frequency bands below fmin and above fmax (int)
        """
        Scale.__init__(self, bnds+beyond*2)
        lfmin = np.log2(fmin)
        lfmax = np.log2(fmax)
        odiv = (lfmax-lfmin)/(bnds-1)
        lfmin_ = lfmin-odiv*beyond
        lfmax_ = lfmax+odiv*beyond
        self.fmin = 2**lfmin_
        self.fmax = 2**lfmax_
        self.pow2n = 2**odiv
        self.q = np.sqrt(self.pow2n)/(self.pow2n-1.)/2.
        
    def F(self, bnd=None):
        return self.fmin*self.pow2n**(bnd if bnd is not None else np.arange(self.bnds))
    
    def Q(self, bnd=None):
        return self.q
    

class LinScale(Scale):
    def __init__(self, fmin, fmax, bnds, beyond=0):
        """
        @param fmin: minimum frequency (Hz)
        @param fmax: maximum frequency (Hz)
        @param bnds: number of frequency bands (int)
        @param beyond: number of frequency bands below fmin and above fmax (int)
        """
        self.df = float(fmax-fmin)/(bnds-1)
        Scale.__init__(self, bnds+beyond*2)
        self.fmin = float(fmin)-self.df*beyond
        if self.fmin <= 0:
            raise ValueError("Frequencies must be > 0.")
        self.fmax = float(fmax)+self.df*beyond

    def F(self, bnd=None):
        return (bnd if bnd is not None else np.arange(self.bnds))*self.df+self.fmin

    def Q(self, bnd=None):
        return self.F(bnd)/(self.df*2)


def hz2mel(f):
    "\cite{shannon:2003}"
    return np.log10(f/700.+1.)*2595.


def mel2hz(m):
    "\cite{shannon:2003}"
    return (np.power(10.,m/2595.)-1.)*700.


class VarQScale(Scale):
    """
    Variable Q-factor logarithmic frequency scale.

    Concatenates multiple log-scale zones with different bins-per-octave (BPO),
    so that time resolution adapts with frequency:

    * Low frequencies (bass drum territory) get a small BPO → short windows →
      good temporal precision at the cost of some frequency resolution.
    * High frequencies (fingerstyle, transient detail) get a large BPO → the
      window is already short at those frequencies so the high BPO is free.

    The Q-factor within each zone is derived from its BPO exactly as LogScale
    does: ``Q = √(2^(1/bpo)) / (2^(1/bpo) − 1) / 2``.

    Parameters
    ----------
    fmin : float
        Minimum centre frequency (Hz).
    fmax : float
        Maximum centre frequency (Hz).
    zones : sequence of (upper_freq_hz, bpo) pairs
        Each pair defines the upper bound and BPO for one zone.  The first
        zone starts at *fmin*; subsequent zones start where the previous one
        ended.  The last zone should cover up to *fmax*.  Default zones give
        roughly ERB-like coverage::

            (   80 Hz,  4 bpo)  → Q ≈  2.9,  window ≈ 18 ms at 40 Hz
            (  320 Hz,  8 bpo)  → Q ≈  5.7,  window ≈  9 ms at 160 Hz
            ( 2000 Hz, 24 bpo)  → Q ≈ 17.3,  window ≈  2 ms at 1 kHz
            (16000 Hz, 38 bpo)  → Q ≈ 27.3,  window ≈  1 ms at 8 kHz
    """

    DEFAULT_ZONES = (
        (   80.0,  4),
        (  320.0,  8),
        ( 2000.0, 24),
        (16000.0, 38),
    )

    def __init__(self, fmin, fmax, zones=None):
        if zones is None:
            zones = self.DEFAULT_ZONES

        all_freqs = []
        all_qs    = []
        lo        = float(fmin)
        fmax      = float(fmax)

        for (hi, bpo) in zones:
            hi = min(float(hi), fmax)
            if lo >= fmax:
                break
            pow2n  = 2.0 ** (1.0 / bpo)
            q_zone = np.sqrt(pow2n) / (pow2n - 1.0) / 2.0
            # Number of bins to reach hi from lo at this BPO
            n = max(1, int(np.ceil(np.log2(hi / lo) * bpo)))
            for i in range(n):
                f = lo * 2.0 ** (float(i) / bpo)
                if f >= fmax:
                    break
                all_freqs.append(f)
                all_qs.append(q_zone)
            # Advance lo to the true endpoint of this zone's last bin
            lo = lo * 2.0 ** (float(n) / bpo)

        self._freqs = np.array(all_freqs, dtype=float)
        self._qs    = np.array(all_qs,    dtype=float)
        Scale.__init__(self, len(self._freqs))

    def F(self, bnd=None):
        if bnd is None:
            bnd = np.arange(self.bnds)
        return self._freqs[bnd]

    def Q(self, bnd=None):
        if bnd is None:
            bnd = np.arange(self.bnds)
        return self._qs[bnd]


class MelScale(Scale):
    def __init__(self, fmin, fmax, bnds, beyond=0):
        """
        @param fmin: minimum frequency (Hz)
        @param fmax: maximum frequency (Hz)
        @param bnds: number of frequency bands (int)
        @param beyond: number of frequency bands below fmin and above fmax (int)
        """
        mmin = hz2mel(fmin)
        mmax = hz2mel(fmax)
        Scale.__init__(self, bnds+beyond*2)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.mbnd = (mmax-mmin)/(bnds-1)  # mels per band
        self.mmin = mmin-self.mbnd*beyond
        self.mmax = mmax+self.mbnd*beyond
        
    def F(self, bnd=None):
        if bnd is None:
            bnd = np.arange(self.bnds)
        return mel2hz(bnd*self.mbnd+self.mmin)

    def Q1(self, bnd=None): # obviously not exact
        if bnd is None:
            bnd = np.arange(self.bnds)
        mel = bnd*self.mbnd+self.mmin
        odivs = (np.exp(mel/-1127.)-1.)*(-781.177/self.mbnd)
        pow2n = np.power(2, 1./odivs)
        return np.sqrt(pow2n)/(pow2n-1.)/2.
    