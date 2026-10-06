//go:build external_native

package fastlowess

/*
#cgo CFLAGS: -I${SRCDIR}/../include -I${SRCDIR}/include
#cgo linux LDFLAGS: -L${SRCDIR}/../../../target/release-c -lfastlowess_go -lm -ldl -lpthread
#cgo darwin LDFLAGS: -L${SRCDIR}/../../../target/release-c -lfastlowess_go
#cgo windows,amd64 LDFLAGS: -L${SRCDIR}/../../../target/x86_64-pc-windows-gnu/release-c -lfastlowess_go -lws2_32 -luserenv -lbcrypt -lntdll -lpthread
#cgo windows,arm64 LDFLAGS: -static -L${SRCDIR}/../../../target/aarch64-pc-windows-gnullvm/release-c -lfastlowess_go -lws2_32 -luserenv -lbcrypt -lntdll -lpthread
*/
import "C"
