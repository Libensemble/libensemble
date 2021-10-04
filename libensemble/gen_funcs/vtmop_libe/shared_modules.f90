! This file (shared_modules.f95) contains modules REAL_PRECISION.

MODULE REAL_PRECISION  ! From HOMPACK90.
  ! This is for 64-bit arithmetic.
  INTEGER, PARAMETER:: R8=SELECTED_REAL_KIND(13)
END MODULE REAL_PRECISION
