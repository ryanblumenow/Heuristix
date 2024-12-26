import streamlit as st

# HTML code for embedding
html_code = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Bars with Images</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column; /* Stack chart and logo vertically */
            justify-content: center;
            align-items: center;
            margin: 0;
            height: 100vh;
            background-color: #f4f4f4;
        }

        .chart-container {
            display: flex;
            align-items: flex-end;
            gap: 20px;
            position: relative;
        }

        .bar {
            width: 100px;
            position: relative;
            transition: transform 0.3s ease;
            cursor: pointer;
        }

        .bar img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            border-radius: 10px;
        }

        .bar:hover {
            transform: scale(1.1);
        }

        .tooltip {
            position: absolute;
            background: #fff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
            font-size: 18px;
            font-weight: bold;
            color: #333;
            width: 300px;
            text-align: center;
            display: none;
            pointer-events: none;
            white-space: normal;
            z-index: 10;
        }

        .logo-container {
            margin-top: 20px; /* Add space between bars and logo */
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .logo-container img {
            width: 390px; /* Adjust size of the logo */
            height: auto;
            object-fit: contain;
        }


    </style>
</head>
<body>

<div class="chart-container">
    <div class="bar" style="height: 200px;" data-description="This is Analytical Insights, focusing on data analysis.">
        <img src="https://i.postimg.cc/67YnrDNB/edalbl.jpg" alt="Analytical Insights">
    </div>
    <div class="bar" style="height: 300px;" data-description="This is Domain Expertise, emphasizing industry knowledge.">
        <img src="https://i.postimg.cc/68xrtr0x/correllbl.jpg" alt="Domain Expertise">
    </div>
    <div class="bar" style="height: 400px;" data-description="This is Artificial Intelligence, leveraging advanced algorithms.">
        <img src="https://i.postimg.cc/94TszY3P/hyptestinglbl.jpg" alt="Artificial Intelligence">
    </div>
    <div class="tooltip" id="tooltip"></div>
</div>

<div class="logo-container">
    <img src="https://i.postimg.cc/TwhQX0JV/Heuristix-logo.png" alt="Heuristix">
</div>


<script>
    const bars = document.querySelectorAll('.bar');
    const tooltip = document.getElementById('tooltip');

    bars.forEach(bar => {
        bar.addEventListener('mouseenter', (e) => {
            const description = bar.getAttribute('data-description');
            tooltip.textContent = description;
            tooltip.style.display = 'block';

            // Position tooltip directly over the bar
            const rect = bar.getBoundingClientRect();
            tooltip.style.left = `${rect.left + rect.width / 2 - tooltip.offsetWidth / 2 - 500}px`;
            tooltip.style.top = `${rect.top - tooltip.offsetHeight - 10}px`; // Slightly above the bar
        });

        bar.addEventListener('mouseleave', () => {
            tooltip.style.display = 'none';
        });
    });
</script>

</body>
</html>

"""

# Embed in Streamlit
st.components.v1.html(html_code, height=700, scrolling=False)