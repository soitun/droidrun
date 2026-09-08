package ai.mobilerun.tapfixture;

import android.app.Activity;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.os.Bundle;
import android.view.Gravity;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.PopupWindow;
import android.widget.TextView;

public class MainActivity extends Activity {
    private TextView result;
    private LinearLayout root;
    private PopupWindow popup;
    private int sequence = 0;

    private int dp(int value) {
        return (int) (getResources().getDisplayMetrics().density * value);
    }

    private void hit(String name) {
        result.setText("RESULT:" + name + ":" + (++sequence));
    }

    private TextView label(String text) {
        TextView view = new TextView(this);
        view.setText(text);
        view.setTextSize(17);
        view.setGravity(Gravity.CENTER);
        return view;
    }

    private Button button(String text) {
        Button button = new Button(this);
        button.setText(text);
        button.setAllCaps(false);
        // Pressing a button must not animate its elevation above its siblings.
        button.setStateListAnimator(null);
        button.setElevation(0);
        return button;
    }

    private FrameLayout row(String description) {
        FrameLayout frame = new FrameLayout(this);
        frame.setContentDescription(description);
        frame.setBackgroundColor(Color.rgb(225, 235, 245));
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(-1, dp(90));
        params.setMargins(dp(8), dp(4), dp(8), dp(4));
        root.addView(frame, params);
        return frame;
    }

    @Override
    public void onCreate(Bundle saved) {
        super.onCreate(saved);
        root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(Color.WHITE);
        setContentView(root);
        result = label("RESULT:none:0");
        root.addView(result, new LinearLayout.LayoutParams(-1, dp(55)));

        FrameLayout decorative = row("Decorative row");
        LinearLayout contents = new LinearLayout(this);
        TextView icon = label("Icon");
        icon.setBackgroundColor(Color.LTGRAY);
        contents.addView(icon, new LinearLayout.LayoutParams(dp(80), -1));
        contents.addView(label("Row label"), new LinearLayout.LayoutParams(0, -1, 1));
        decorative.addView(contents, new FrameLayout.LayoutParams(-1, -1));
        decorative.setOnClickListener(view -> hit("decorative-parent"));

        FrameLayout independent = row("Independent row");
        independent.addView(label("Parent area"),
                new FrameLayout.LayoutParams(dp(200), -1, Gravity.LEFT));
        Button child = button("Child action");
        independent.addView(child,
                new FrameLayout.LayoutParams(dp(125), -1, Gravity.RIGHT));
        independent.setOnClickListener(view -> hit("independent-parent"));
        child.setOnClickListener(view -> hit("independent-child"));

        FrameLayout overlap = row("Overlap section");
        Button target = button("Partly covered target");
        overlap.addView(target, new FrameLayout.LayoutParams(-1, -1));
        target.setOnClickListener(view -> hit("overlap-target"));
        Button overlay = button("Overlay action");
        overlap.addView(overlay,
                new FrameLayout.LayoutParams(dp(170), -1, Gravity.CENTER));
        overlay.setOnClickListener(view -> hit("overlap-overlay"));

        Button focus = button("Show focusable popup");
        root.addView(focus);
        focus.setOnClickListener(view -> showPopup(true));
        Button nonfocus = button("Show nonfocusable popup");
        root.addView(nonfocus);
        nonfocus.setOnClickListener(view -> showPopup(false));
    }

    private void showPopup(boolean focusable) {
        if (popup != null) {
            popup.dismiss();
        }
        LinearLayout content = new LinearLayout(this);
        content.setOrientation(LinearLayout.VERTICAL);
        content.setBackgroundColor(Color.rgb(255, 235, 205));
        content.addView(label(focusable ? "Focusable popup" : "Nonfocusable popup"));
        Button action = button("Popup action");
        content.addView(action);
        action.setOnClickListener(view -> {
            hit(focusable ? "focusable-popup" : "nonfocusable-popup");
            popup.dismiss();
        });
        popup = new PopupWindow(content, dp(270), dp(150), focusable);
        popup.setBackgroundDrawable(new ColorDrawable(Color.WHITE));
        popup.setOutsideTouchable(true);
        popup.setTouchable(true);
        popup.setElevation(dp(8));
        popup.showAtLocation(root, Gravity.CENTER, 0, 0);
    }
}
