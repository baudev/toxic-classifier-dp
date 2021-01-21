// DOM elements
const results_dom = document.getElementById("results");
const submit_button_dom = document.getElementById("submit_button");
const sentence_input_dom = document.getElementById("sentence_input");
const loader_dom = document.getElementById("loader");
const classes_dom = document.getElementById("classes");
const estimated_probabilities_dom = document.getElementById("estimated_probabilities");
const estimated_probabilities_classe_names = document.getElementById("estimated_probabilities_classe_names");
const ctx_dom = document.getElementById('chart').getContext('2d');

// Variables
const classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"];
let vector_size = 0
let model = undefined;
let vocabulary = [];


submit_button_dom.onclick = async () => {
    // we disable the button and the input
    submit_button_dom.classList.add("disabled");
    sentence_input_dom.classList.add("disabled");

    // we show the loader
    loader_dom.style.display = "block";

    // load tensorflow model
    if (model === undefined) {
        model = await tf.loadLayersModel('https://raw.githubusercontent.com/baudev/toxic-classifier-dp/develop/model_js/model.json', strict = false)
        // input shape for the vector size
        vector_size = model.layers[0].inboundNodes[0].inputShapes[0][1]
    }

    // get the vector corresponding to the text input
    let vector = await getVectorOfSentence(sentence_input_dom.value, vector_size);
    // create the tensorflow tensor
    let x = tf.tensor([vector])
    // calculate the predictions
    let probabilities = await model.predict(x).data()

    // we show the results
    loader_dom.style.display = "none";
    results_dom.style.display = "block";

    // estimated probabilities and classes
    estimated_probabilities_dom.innerHTML = "";
    classes_dom.innerHTML = "";
    estimated_probabilities_classe_names.innerHTML = "";

    // shows the probabilities
    _.forEach(probabilities, (probability) => {
        let td = document.createElement("td");
        td.textContent = Number.parseFloat(probability).toFixed(2);
        estimated_probabilities_dom.appendChild(td);
    })

    // shows the classes
    let counter = 0;
    for (let i = 0; i < classes.length; i++) {
        let th = document.createElement("th");
        th.textContent = classes[i];
        estimated_probabilities_classe_names.appendChild(th);

        if (probabilities[i] > 0.5) {
            counter++;
            let p = document.createElement("p");
            p.textContent = classes[i];
            classes_dom.appendChild(p);
        }
    }

    if(counter === 0) {
        // the text is not toxic
        M.toast({html: 'Congratulations, your sentence is not toxic!', classes: 'green'});
    }

    // draws the chart
    const chart = new Chart(ctx_dom, {
        type: 'bar',
        data: {
            labels: classes,
            datasets: [{
                backgroundColor: '#ff9800',
                data: probabilities
            }]
        },

        // Configuration
        options: {
            scales: {
                yAxes: [{
                    display: true,
                    ticks: {
                        beginAtZero: true,
                        max: 1
                    }
                }]
            },
            legend: {
                display: false
            },
        }
    });

    // Enables the text input and the button
    submit_button_dom.classList.remove("disabled");
    sentence_input_dom.classList.remove("disabled");

}

/**
 * Preprocess the string
 * @param string
 * @returns the string preprocessed
 */
function preprocess(string) {
    string = string.toLowerCase();
    string = string.replace(/[^\sa-z]/gm, '')
    string = string.replace(/ {2,}/gm, ' ')
    return string.trim();
}

/**
 * Reads JSON file
 * @param file
 * @returns {Promise<JSON>}
 */
function readJSONFile(file) {
    return new Promise((resolve => {
        const rawFile = new XMLHttpRequest();
        rawFile.overrideMimeType("application/json");
        rawFile.open("GET", file, true);
        rawFile.onreadystatechange = function() {
            if (rawFile.readyState === 4 && rawFile.status === 200) {
                resolve(JSON.parse(rawFile.responseText));
            }
        }
        rawFile.send(null);
    }))
}

/**
 * Gets the corresponding vocabulary index of the word
 * @param word
 * @returns {Promise<int>}
 */
async function getIndexOfWord(word) {
    if (vocabulary.length === 0) {
        // read vocab if needed
        vocabulary = await readJSONFile("https://raw.githubusercontent.com/baudev/toxic-classifier-dp/develop/model_js/vocabulary.json");
    }
    let index = await _.findIndex(vocabulary, function(o) {
        return o === word;
    });
    return index === -1 ? 1 : index;
}

/**
 * Returns the corresponding vector of the sentence
 * @param sentence
 * @param pad_size
 * @returns {Promise<Array>}
 */
async function getVectorOfSentence(sentence, pad_size) {
    sentence = preprocess(sentence);
    let array = sentence.split(" ");
    array = _.map(array, function getIndex(value) {
        return getIndexOfWord(value);
    })
    array = await Promise.all(array)
    array = _.slice(_.assign(_.fill(new Array(pad_size), 0), array), 0, pad_size)
    return array;
}