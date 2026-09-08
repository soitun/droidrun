# Android indexed-tap fixture

Use an Android emulator with Portal 0.7.25 or newer. This isolated app shows
which click handler actually ran (`RESULT:<handler>:<sequence>`). Its overlapping
buttons have no elevation animation, so pressing one does not reorder them.

## Build and install

From this directory, with JDK 17, Android SDK platform 36 and build-tools 36.1.0:

```bash
set -e
SDK="${ANDROID_HOME:?Set ANDROID_HOME to your Android SDK}"
BT="$SDK/build-tools/36.1.0"
BUILD="$(mktemp -d)"
mkdir "$BUILD/classes" "$BUILD/dex"
"$BT/aapt2" link -o "$BUILD/unsigned.apk" --manifest AndroidManifest.xml \
  -I "$SDK/platforms/android-36/android.jar"
javac --release 8 -classpath "$SDK/platforms/android-36/android.jar" \
  -d "$BUILD/classes" MainActivity.java
"$BT/d8" --lib "$SDK/platforms/android-36/android.jar" --output "$BUILD/dex" \
  "$BUILD/classes/ai/mobilerun/tapfixture/"*.class
zip -j "$BUILD/unsigned.apk" "$BUILD/dex/classes.dex"
"$BT/zipalign" -f 4 "$BUILD/unsigned.apk" "$BUILD/aligned.apk"
keytool -genkeypair -keystore "$BUILD/debug.keystore" -storepass android \
  -keypass android -alias androiddebugkey -keyalg RSA -validity 30 \
  -dname "CN=Tap fixture"
"$BT/apksigner" sign --ks "$BUILD/debug.keystore" --ks-pass pass:android \
  --out "$BUILD/fixture.apk" "$BUILD/aligned.apk"
"$SDK/platform-tools/adb" -s emulator-5554 install "$BUILD/fixture.apk"
```

Use a Python environment with this repository and `mobilerun-core[local]` installed.
`validate.py` uses the actual Android state provider, concise filter, indexed
formatter, `click(index)` action, and a device-tap adapter through Mobilerun core.
It saves the raw trees, screenshots, selected formatted node, dispatched
coordinates, and observed outcome. A successful tool response alone does not pass.

```bash
python validate.py --serial emulator-5554 --output /tmp/tap-launch \
  --launch ai.mobilerun.tapfixture
python validate.py --serial emulator-5554 --output /tmp/tap-target \
  --target 'Partly covered target' --expect 'RESULT:overlap-target'
```

Run the same commands with unmodified `main` and the fix installed in separate
Python environments. Keep the device, Portal, and fixture identical.

| Target | Required outcome |
| --- | --- |
| Decorative row | `RESULT:decorative-parent` |
| Independent row | `RESULT:independent-parent` |
| Child action | `RESULT:independent-child` |
| Partly covered target | `RESULT:overlap-target` |
| Overlay action | `RESULT:overlap-overlay` |
| Show focusable popup, then Popup action | `Focusable popup`, then `RESULT:focusable-popup` |
| Show nonfocusable popup, then Popup action | `Nonfocusable popup`, then `RESULT:nonfocusable-popup` |

The decorative row's non-clickable icon and label fill the entire row. The
independent child covers the right side, leaving the parent's center unobstructed.
The overlay covers the target's center but leaves side strips available.

## Investigation outcome

On Android 16 / Portal 0.7.25 at 1080×2400, baseline `7405272` selected index 10
(`Partly covered target`, bounds `21,729,1059,965`) and tapped `(540,847)`.
The tool returned success, but `RESULT:overlap-overlay` appeared. The overlay's
bounds were `317,729,763,965`, with a higher drawing order in the same window.
The fix selected the same index, tapped `(169,847)`, and produced
`RESULT:overlap-target`.

The child-filled row, independent child, and popup actions worked. This does not
support #435's claimed failure of ordinary clicks on child-filled containers:
`get_clear_point()` is dormant in the normal action path, and changing it would
not fix the observed sibling obstruction.

Baseline also opened Android Settings → Apps by clicking the parent row at
index 17 / `(540,1071)`. A bounded `MobileAgent` run using an existing OpenAI OAuth
profile and `gpt-5.5` clicked `Child action` (index 8); a separate state read
confirmed `RESULT:independent-child:4`, advancing the previous counter by one.

The fix only considers direct, touchable siblings with a strictly higher known
drawing order in the same window. It preserves ordinary center/stealth taps when
unobstructed and rejects a target fully covered by known blockers. It does not
infer global order from accessibility indices or consider all descendants safe.
Portal can attach a non-focusable popup's separate window as a child of the main
root. Cross-window occlusion, different-parent overlays, parent interception,
and unavailable/ambiguous ordering remain outside this reproduction and fix.
