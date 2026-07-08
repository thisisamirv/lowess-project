// We need to forward routine registration from C to Rust
// to avoid the linker removing the static library.
//
// Note: R's Boolean.h in R 4.5.x uses a C23 enum underlying type feature
// which triggers a -Wpedantic warning in older C standards. To avoid a
// CRAN NOTE about suppressing diagnostics with pragmas, we pre-define
// the Rboolean type and set its header guard to prevent R's version
// from being loaded. We also include <stdbool.h> as R's headers now
// expect 'bool' to be defined.

#ifndef R_EXT_BOOLEAN_H_
#define R_EXT_BOOLEAN_H_
#include <stdbool.h>
#undef FALSE
#undef TRUE
typedef enum { FALSE = 0, TRUE } Rboolean;
#endif

#include <R.h>
#include <Rinternals.h>

extern void R_init_rfastlowess_extendr(DllInfo *dll);
void register_extendr_panic_hook(void); // NOLINT(readability-identifier-naming)

void R_init_rfastlowess(DllInfo *dll) { // NOLINT(misc-use-internal-linkage)
  register_extendr_panic_hook();
  R_init_rfastlowess_extendr(dll);
}
