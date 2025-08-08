import streamlit as st
import streamlit.components.v1 as components
import json
from streamlit_javascript import st_javascript

def render_donut_chart(data, donut_radius=150, thickness_ratio=0.3, key="donut_chart"):
    donut_width = int(donut_radius * thickness_ratio)
    canvas_padding = 5
    
    # --- ✨ MODIFIED: Height is now based on radius, not diameter ---
    width = donut_radius * 2 + canvas_padding * 2
    height = donut_radius + canvas_padding + 20 # Added 20px for labels
    
    js_data = json.dumps(data)

# ... inside render_donut_chart ...

    html_code = f"""
    <div style="display: flex; justify-content: center; align-items: center;">
      <canvas id="donutChart" width="{width}" height="{height}" style="cursor: pointer;"></canvas>
    </div>
    <input id="valueInput" type="number" step="1" style="
      position: absolute; display: none; z-index: 9999;
      font-size: 14px; padding: 2px; width: 60px;
      background: transparent; color: white; border: 1px solid white;
    "/>
    <script>
      const canvas = document.getElementById("donutChart");
      const ctx = canvas.getContext("2d");
      const input = document.getElementById("valueInput");
      let data = {js_data};
      const baseRadius = {donut_radius};
      const donutWidth = {donut_width};
      const centerX = canvas.width / 2;
      const centerY = canvas.height - 20;
      let hoveredIndex = -1;
      let canvasRect = null;
      let segments = [];

      function drawDonut() {{
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const total = data.reduce((sum, d) => sum + d.value, 0);
        let startAngle = Math.PI;
        segments = [];

        data.forEach((d, index) => {{
          const angle = (d.value / total) * Math.PI;
          const endAngle = startAngle + angle;
          const radius = index === hoveredIndex ? baseRadius + 10 : baseRadius;

          ctx.beginPath();
          ctx.moveTo(centerX, centerY);
          ctx.arc(centerX, centerY, radius, startAngle, endAngle);
          ctx.closePath();
          ctx.fillStyle = d.color;
          ctx.fill();

          segments.push({{ label: d.label, startAngle: startAngle, endAngle: endAngle }});

          const midAngle = (startAngle + endAngle) / 2;
          const textRadius = radius - donutWidth / 2;
          const labelX = centerX + textRadius * Math.cos(midAngle);
          const labelY = centerY + textRadius * Math.sin(midAngle);

          ctx.save();
          ctx.translate(labelX, labelY);
          ctx.fillStyle = "#ffffff";
          ctx.font = "bold 14px sans-serif";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(`${{d.label}}: ${{d.value.toFixed(2)}}`, 0, 0);
          ctx.restore();

          startAngle = endAngle;
        }});

        ctx.save();
        ctx.globalCompositeOperation = "destination-out";
        ctx.beginPath();
        ctx.arc(centerX, centerY, baseRadius - donutWidth, 0, 2 * Math.PI);
        ctx.fill();
        ctx.restore();
      }}

      drawDonut();

      canvas.addEventListener("mousemove", function(event) {{
        canvasRect = canvas.getBoundingClientRect();
        const x = event.clientX - canvasRect.left - centerX;
        const y = event.clientY - canvasRect.top - centerY;
        const distance = Math.sqrt(x * x + y * y);
        let currentHover = -1;

        if (distance < baseRadius + 10 && distance > (baseRadius - donutWidth)) {{
          let angle = Math.atan2(y, x);
          if (angle < 0) angle += 2 * Math.PI;

          for (let i = 0; i < segments.length; i++) {{
            if (angle >= segments[i].startAngle && angle <= segments[i].endAngle) {{
              currentHover = i;
              break;
            }}
          }}
        }}

        if (hoveredIndex !== currentHover) {{
          hoveredIndex = currentHover;
          drawDonut();
        }}
      }});

      canvas.addEventListener("mouseleave", () => {{
        hoveredIndex = -1;
        drawDonut();
      }});

      canvas.addEventListener("click", function(event) {{
        canvasRect = canvas.getBoundingClientRect();
        const x = event.clientX - canvasRect.left - centerX;
        const y = event.clientY - canvasRect.top - centerY;
        const distance = Math.sqrt(x * x + y * y);

        if (distance < baseRadius && distance > (baseRadius - donutWidth)) {{
          let angle = Math.atan2(y, x);
          if (angle > 0) angle -= 2 * Math.PI;

          for (let i = 0; i < segments.length; i++) {{
            const s = segments[i];
            if (angle >= s.startAngle - 2 * Math.PI && angle <= s.endAngle - 2 * Math.PI) {{
              const canvasX = centerX + (baseRadius - donutWidth / 2) * Math.cos((s.startAngle + s.endAngle) / 2);
              const canvasY = centerY + (baseRadius - donutWidth / 2) * Math.sin((s.startAngle + s.endAngle) / 2);
              const screenX = canvasRect.left + canvasX;
              const screenY = canvasRect.top + canvasY;

              input.style.left = `${{screenX}}px`;
              input.style.top = `${{screenY}}px`;
              input.value = data[i].value;
              input.style.display = "block";
              input.focus();
              input.select();

              input.onblur = () => {{
                const parsed = parseFloat(input.value);
                if (!isNaN(parsed) && parsed >= 0) {{
                  data[i].value = parsed;
                  drawDonut();
                  localStorage.setItem("donut_data", JSON.stringify(data));
                }}
                input.style.display = "none";
              }};
              input.onkeydown = (e) => {{ if (e.key === "Enter") input.blur(); }};
              break;
            }}
          }}
        }}
      }});
    </script>
    """

    centered_html = f"""
    <div style="display: flex; justify-content: center; align-items: center; height: {height + 20}px;">
        {html_code}
    </div>
    """
    components.html(html_code, height=height + 20)

# # ----------------------
# # Streamlit Setup
# # ----------------------

# st.set_page_config(layout="centered")
# default_data = [
#     {"label": "Petrol", "value": 20, "color": "#f94144"},
#     {"label": "Diesel", "value": 25, "color": "#f3722c"},
#     {"label": "Biofuel", "value": 15, "color": "#90be6d"},
#     {"label": "Ethanol", "value": 30, "color": "#577590"},
#     {"label": "Additive", "value": 10, "color": "#f8961e"},
# ]

# # --- ✨ MODIFIED: Use columns to center the chart ---
# left_col, center_col, right_col = st.columns([1, 4, 1]) # Give the center column more space
# with center_col:
#     render_donut_chart(default_data, donut_radius=200, thickness_ratio=0.3)


# # 🔁 Sync JS → Streamlit
# result = st_javascript(
#     """
#     const raw = localStorage.getItem("donut_data");
#     try {
#       const parsed = JSON.parse(raw);
#       if (Array.isArray(parsed)) return parsed;
#     } catch (e) {}
#     return null;
#     """,
#     key="donut_sync"
# )

# if result:
#     st.subheader("📥 Updated Data from Donut:")
#     st.json(result)
# else:
#     st.info("ℹ️ Interact with the chart and values will appear here.")